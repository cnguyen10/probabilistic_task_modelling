"""
python3 gmm_vi.py --datasource=omniglot-py --label --img-suffix=.png --ds-root-folder=../datasets --K=8 --k-shot=10 --v-shot=10 --min-num-cls=5 --max-num-cls=20 --minibatch=20 --nz=64 --nf=8 --lr=2e-4 --gmm-lr=1e-3 --alpha=1.1 --KL-weight=0.1 --num-epochs=100 --num-episodes-per-epoch=10000 --num-em-steps=5 --resume-epoch=0

python3 gmm_vi.py --datasource=miniImageNet_64 --label --img-suffix=.png --ds-root-folder=../datasets --K=8 --k-shot=16 --v-shot=16 --min-num-cls=5 --max-num-cls=10 --minibatch=20 --nz=128 --nf=32 --lr=2e-4 --gmm-lr=1e-3 --alpha=1.1 --KL-weight=0.05 --num-epochs=100 --num-episodes-per-epoch=10000 --num-em-steps=5 --resume-epoch=0
"""
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import argparse
import typing

from encoder_decoder import Encoder, Decoder
import EpisodeGenerator

from utils import train_val_split, weights_init, euclidean_distance, get_cls_prototypes, get_reconstruction_loss, batch_trace, dirichlet_expected_log

# -------------------------------------------------------------------------------------------------
# Setup input parser
# -------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Setup variables')

parser.add_argument('--datasource', type=str, default='omniglot-py', help='Data set: omniglot-py, ImageNet64')
parser.add_argument('--img-suffix', type=str, default='.png', help='')
parser.add_argument('--ds-root-folder', type=str, default='../datasets/', help='Root folder containing a folder of the data set')
parser.add_argument('--logdir-root', type=str, default='/media/n10/Data/gmm_vi', help='Directory to store logs and models')

# the purpose of --label is to include/exclude the NLL -ln p(y | x) corresponding to
# the classification loss in Eq (15)
parser.add_argument('--label', dest='label', action='store_true')
parser.add_argument('--no-label', dest='label', action='store_false')
parser.set_defaults(label=True)

parser.add_argument('--alpha', type=float, default=1.1, help='Dirichlet prior')

parser.add_argument('--K', type=int, default=8, help='Number of components in the Gaussian mixture model')
parser.add_argument('--num-em-steps', type=int, default=10, help='Number of EM iterations for an episode')

parser.add_argument('--k-shot', type=int, default=10, help='Number of labeled samples per class')
parser.add_argument('--v-shot', type=int, default=10, help='Number of labeled samples per class')
parser.add_argument('--min-num-cls', type=int, default=5, help='Minimum number of classes an episode can contain')
parser.add_argument('--max-num-cls', type=int, default=5, help='Maximum number of classes an episode can contain')

parser.add_argument('--minibatch', type=int, default=10, help='Mini-batch size or the number of episode used for a meta update')
parser.add_argument('--nz', type=int, default=64, help='Dimension of latent representation, 64 for Omniglot and 256 for ImageNet')

parser.add_argument('--nf', type=int, default=16, help='The based number of channels used in the VAE')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the VAE')
parser.add_argument('--gmm-lr', type=float, default=1e-3, help='Learning rate for GMM')
parser.add_argument('--KL-weight', type=float, default=0.01, help='Re-weighting factor of KL divergence terms when deriving lower-bounds for VAE and LDA')

parser.add_argument('--resume-epoch', type=int, default=0, help='Epoch id to resume learning or perform testing')

parser.add_argument('--num-episodes-per-epoch', type=int, default=10000, help='Number of episodes in each epoch')
parser.add_argument('--num-epochs', type=int, default=1, help='How many \'epochs\' are going to be?')

parser.add_argument('--train', dest='train_flag', action='store_true')
parser.add_argument('--test', dest='train_flag', action='store_false')
parser.set_defaults(train_flag=True)

args = parser.parse_args()

config = {}
for key in args.__dict__:
    config[key] = args.__dict__[key]
# ----------------------------------------------------------------------
# Setup CPU or GPU
# ----------------------------------------------------------------------
gpu_id = 0
device = torch.device('cuda:{0:d}'.format(gpu_id) \
    if torch.cuda.is_available() else torch.device('cpu'))

# ----------------------------------------------------------------------
# Setup parameters
# ----------------------------------------------------------------------
# directory storing files
config['logdir'] = os.path.join(
    config['logdir_root'],
    '{0:s}_nz{1:d}_nf{2:d}_K{3:d}_alpha{4:.1f}_EM{5:d}_{6:s}'.format(
        config['datasource'],
        config['nz'],
        config['nf'],
        config['K'],
        config['alpha'],
        config['num_em_steps'],
        'label' if config['label'] else 'unlabel'
    )
)
if not os.path.exists(config['logdir']):
    from pathlib import Path
    Path(config['logdir']).mkdir(parents=True, exist_ok=True)
print('Log at: {}'.format(config['logdir']))

config['ds_folder'] = os.path.join(config['ds_root_folder'], config['datasource'])

if config['datasource'] in ['omniglot-py']:
    episode_generator = EpisodeGenerator.OmniglotLoader(
        root=config['ds_folder'],
        train_subset=config['train_flag'],
        suffix=config['img_suffix'],
        load_images=True,
        k_shot=config['k_shot'] + config['v_shot'],
        min_num_cls=config['min_num_cls'],
        max_num_cls=config['max_num_cls']
    )
    img_size = (1, 64, 64)
elif config['datasource'] in ['miniImageNet_64', 'ImageNet64']:
    episode_generator = EpisodeGenerator.ImageFolderGenerator(
        root=config['ds_folder'],
        train_subset=config['train_flag'],
        k_shot=config['k_shot'] + config['v_shot'],
        load_images=False,
        suffix=config['img_suffix']
    )
    img_size = (3, 64, 64)
else:
    raise NotImplementedError('Unknown dataset (Properly dataset name does not match)')

alpha = torch.tensor(data=[config['alpha']] * config['K'], device=device)
log_alpha = torch.log(alpha)
print()

# ----------------------------------------------------------------------
# Inference procedures
# ----------------------------------------------------------------------
def train() -> None:
    """Train VAE and Gaussian mixture model"""
    try:
        minibatch_print = np.lcm(config['minibatch'], 200)

        # initialize/load model
        g_components, encoder, decoder, opt = load_model(epoch_id=config['resume_epoch'], lr=config['lr'])
        print(opt)
        print()

        # zero gradient
        opt.zero_grad()

        # tensorboard to monitor
        tb_writer = SummaryWriter(
            log_dir=config['logdir'],
            purge_step=config['resume_epoch'] * config['num_episodes_per_epoch'] // minibatch_print \
                if config['resume_epoch'] > 0 else None
        )

        gmm_accumulate = {'mean': 0., 'cov': 0.}

        for epoch in range(config['resume_epoch'], config['resume_epoch'] + config['num_epochs']):
            episode_count = 0

            vfe_monitor = 0
            reconstruction_monitor = 0
            vfe_gmm_monitor = 0
            loss_cls_monitor = 0
            entropy_loss_monitor = 0.
            kl_loss_monitor = 0.

            while (episode_count < config['num_episodes_per_epoch']):
                gmm = torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=g_components['mean'],
                    covariance_matrix=g_components['cov']
                )

                # --------------------------------------------------
                # Get episode data from episode generator
                # --------------------------------------------------
                X = episode_generator.generate_episode() # Note: X = (C, k_shot, channels, d, d) = List[List[Tensor]]

                # train - val split
                xt, yt, xv, yv = train_val_split(X=X, k_shot=config['k_shot'], shuffle=True)

                # convert to torch tensors
                x_t = torch.tensor(xt, dtype=torch.float, device=device) # (C * k_shot, nc, H, W)
                y_t = torch.tensor(yt, device=device)
                x_v = torch.tensor(xv, dtype=torch.float, device=device)
                y_v = torch.tensor(yv, device=device)
                
                m_t, s_t = encoder.forward(x=x_t)
                m_v, s_v = encoder.forward(x=x_v)
                
                # distribution of latent embedding
                qu_t = torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=m_t,
                    scale_tril=torch.diag_embed(input=s_t)
                )
                qu_v = torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=m_v,
                    scale_tril=torch.diag_embed(input=s_v)
                )

                # sample latent variable u to calculate reconstruction loss
                u_t = m_t + torch.randn_like(m_t, device=m_t.device) * s_t
                u_v = m_v + torch.randn_like(m_v, device=m_v.device) * s_v
                reconstruction_loss = get_reconstruction_loss(
                    x=x_v,
                    z=u_v,
                    decoder=decoder
                )

                # classification
                if config['label'] == True:
                    prototypes = get_cls_prototypes(x=u_t, y=y_t)
                    distance_matrix = euclidean_distance(matrixN=u_v, matrixM=prototypes)
                    loss_cls = torch.nn.functional.cross_entropy(
                        input=-distance_matrix,
                        target=y_v,
                        reduction='mean'
                    )
                else:
                    loss_cls = torch.tensor(0., device=device)

                # EM with empirical Bayes
                log_r, gamma = E_step(qu=qu_t, gmm=gmm, alpha=alpha)
                with torch.no_grad():
                    new_gmm = M_step(qu=qu_v, log_r=log_r)
                    gmm_accumulate['mean'] += new_gmm.loc
                    gmm_accumulate['cov'] += new_gmm.covariance_matrix

                # ELBO
                vfe_gmm = - elbo(qu=qu_v, gmm=gmm, log_r=log_r, gamma=gamma, alpha=alpha) / len(X)

                # entropy loss of variational distribution qu
                entropy_loss = - torch.mean(input=qu_v.entropy(), dim=0)

                kl_loss = vfe_gmm + entropy_loss
                vfe = reconstruction_loss + (kl_loss * config['KL_weight']) / np.prod(img_size) + loss_cls

                vfe = vfe / config['minibatch']

                # terminate if loss is NaN
                if torch.isnan(vfe):
                    raise ValueError('VFE is NaN.')
                
                # accumulate gradient of meta-parameters
                vfe.backward()

                # --------------------------------------------------
                # monitor losses
                # --------------------------------------------------
                vfe_monitor += vfe.item()
                reconstruction_monitor += reconstruction_loss.item()
                vfe_gmm_monitor += vfe_gmm.item()
                loss_cls_monitor += loss_cls.item()
                entropy_loss_monitor += entropy_loss.item()
                kl_loss_monitor += kl_loss.item()

                episode_count += 1
                # --------------------------------------------------
                # meta-update
                # --------------------------------------------------
                if episode_count % config['minibatch'] == 0:
                    # update VAE parameters
                    torch.nn.utils.clip_grad_norm_(parameters=encoder.parameters(), max_norm=10)
                    torch.nn.utils.clip_grad_norm_(parameters=decoder.parameters(), max_norm=10)
                    opt.step()
                    opt.zero_grad()

                    for key in g_components:
                        gmm_accumulate[key] /= config['minibatch']
                        g_components[key] = g_components[key] + config['gmm_lr'] * (gmm_accumulate[key] - g_components[key])
                    
                    gmm_accumulate = {'mean': 0., 'cov': 0.}

                    # --------------------------------------------------
                    # Monitor results on TensorBoard
                    # --------------------------------------------------
                    if (episode_count % minibatch_print == 0):
                        vfe_monitor = vfe_monitor / minibatch_print * config['minibatch']
                        reconstruction_monitor /= minibatch_print
                        vfe_gmm_monitor /= minibatch_print
                        loss_cls_monitor /= minibatch_print
                        entropy_loss_monitor /= minibatch_print
                        kl_loss_monitor /= minibatch_print

                        with torch.no_grad():
                            log_Nk_monitor = torch.logsumexp(input=log_r, dim=0)
                            Nk_monitor = torch.exp(input=log_Nk_monitor)
                            Nk_monitor = Nk_monitor / torch.sum(input=Nk_monitor) * 100

                        global_step = (epoch * config['num_episodes_per_epoch'] + episode_count) // minibatch_print
                        if config['label']:
                            tb_writer.add_scalar(tag='Loss/Classification', scalar_value=loss_cls_monitor, global_step=global_step)
                        tb_writer.add_scalar(tag='Loss/Reconstruction', scalar_value=reconstruction_monitor, global_step=global_step)
                        tb_writer.add_scalar(tag='Loss/GMM_NLL', scalar_value=vfe_gmm_monitor, global_step=global_step)
                        tb_writer.add_scalar(tag='Loss/Entropy_loss', scalar_value=entropy_loss_monitor, global_step=global_step)
                        tb_writer.add_scalar(tag='Loss/KL_div', scalar_value=kl_loss_monitor, global_step=global_step)
                        tb_writer.add_scalar(tag='Loss/Total',scalar_value=vfe_monitor, global_step=global_step)
                        for k in range(config['K']):
                            tb_writer.add_scalar(tag='Mixture percentage/{0:d}'.format(k), scalar_value=Nk_monitor[k].item(), global_step=global_step)

                        vfe_monitor = 0.
                        reconstruction_monitor = 0.
                        vfe_gmm_monitor = 0.
                        loss_cls_monitor = 0.
                        entropy_loss_monitor = 0.
                        kl_loss_monitor = 0.

            # --------------------------------------------------
            # save model
            # --------------------------------------------------
            checkpoint = {
                'g_components': g_components,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict()
            }
            checkpoint_filename = 'Epoch_{0:d}.pt'.format(epoch + 1)
            torch.save(checkpoint, os.path.join(config['logdir'], checkpoint_filename))
            checkpoint = 0
            print('SAVING parameters into {0:s}'.format(checkpoint_filename))
            print('----------------------------------------\n')
    finally:
        # --------------------------------------------------
        # clean up
        # --------------------------------------------------
        print('\nClose tensorboard summary writer')
        tb_writer.close()
    
    return None

# ----------------------------------------------------------------------
# Auxilliary procedures
# ----------------------------------------------------------------------
def E_step(qu: torch.distributions.MultivariateNormal, gmm: torch.distributions.MultivariateNormal, alpha: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """calculate variational parameters of q(z; r) and q(pi; gamma)

    Args:
        qu: the normal distribution of latent variable u
        g_components: meta Gaussian mixture model
        alpha: the Dirichlet parameter

    Return:
        log_r:
        gamma:
    """
    # initialize gamma
    gamma = torch.ones(config['K'], device=device) * alpha + qu.mean.__len__() / config['K']

    num_em_count = 0
    # dgamma = 1
    # gamma_prev = 0

    # constant values for r
    precision_matrix_k = gmm.precision_matrix.float() # (K, d, d)
    trace_precision_Su = batch_trace(x=torch.matmul(
        input=precision_matrix_k, # (K, d, d)
        other=qu.covariance_matrix[:, None, :, :]) # (N, 1, d, d)
    ) # (N, K)
    log_prob_mu = gmm.log_prob(value=qu.mean[:, None, :]) # (N, K)

    log_r = torch.empty_like(input=log_prob_mu, device=device)

    # while (dgamma > config['gamma_tol']) and (num_em_count < config['num_em_steps']):
    while (num_em_count < config['num_em_steps']):
        # --------------------------------------------------
        # solve for r
        # --------------------------------------------------
        log_pi_tilde = dirichlet_expected_log(concentration=gamma) # (K, )

        log_r_unnormalized = log_pi_tilde - trace_precision_Su / 2 + log_prob_mu # (N, K)
        log_r = torch.log_softmax(input=log_r_unnormalized, dim=1) # (N, K)

        # --------------------------------------------------
        # solve for gamma
        # --------------------------------------------------
        log_Nk = torch.logsumexp(input=log_r, dim=0) # (K, )
        log_gamma = torch.logaddexp(input=log_alpha, other=log_Nk)
        gamma = torch.exp(input=log_gamma)
        
        # with torch.no_grad():
        #     dgamma = torch.mean(input=torch.abs(input=gamma - gamma_prev)).item()
        #     gamma_prev = gamma.detach()
        num_em_count += 1

    return log_r, gamma

def M_step(qu: torch.distributions.MultivariateNormal, log_r: torch.Tensor) -> torch.distributions.MultivariateNormal:
    """
    """
    log_normalized = log_r - torch.logsumexp(input=log_r, dim=0) # (N, K)
    normalized = torch.exp(input=log_normalized) # (N, K)

    new_mean = torch.sum(input=normalized[:, :, None] * qu.mean[:, None, :], dim=0)

    dmean = qu.mean[:, None, :, None] - new_mean[None, :, :, None] # (N, K, d, 1)
    dmean_square = torch.matmul(
        input=dmean,
        other=torch.transpose(input=dmean, dim0=-2, dim1=-1)
    )
    new_cov = torch.sum(
        input=normalized[:, :, None, None] * (qu.covariance_matrix[:, None, :, :] + dmean_square),
        dim=0
    )

    try:
        new_gmm = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=new_mean,
            covariance_matrix=new_cov
        )
    except RuntimeError:
        print('Singularity due to underflow.\nReset means and covariance matrices')
        new_gmm = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.randn(size=(config['K'], config['nz']), device=device),
            scale_tril=torch.diag_embed(input=torch.ones(size=(config['K'], config['nz']), device=device))
        )

    return new_gmm

def elbo(qu: torch.distributions.MultivariateNormal, gmm: torch.distributions.MultivariateNormal, log_r: torch.Tensor, gamma: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    """
    r = torch.exp(input=log_r) #(N, K)
    log_pi_tilde = dirichlet_expected_log(concentration=gamma)

    # --------------------------------------------------
    # E_ln_p_u_z_gmm
    # --------------------------------------------------
    precision_matrix_k = gmm.precision_matrix.float()

    trace_precision_Su = batch_trace(x=torch.matmul(
        input=precision_matrix_k, # (K, d, d)
        other=qu.covariance_matrix[:, None, :, :]) # (N, 1, d, d)
    ) # (N, K)
    log_prob_mu = gmm.log_prob(value=qu.mean[:, None, :]) # (N, K)

    E_ln_pu = torch.sum(input=r * (log_prob_mu - trace_precision_Su / 2))

    # --------------------------------------------------
    # E_ln_pz
    # --------------------------------------------------
    E_ln_pz = torch.sum(input=r * log_pi_tilde)

    # --------------------------------------------------
    # E_ln_q_z
    # --------------------------------------------------
    E_ln_qz = torch.sum(input=r * log_r)

    # --------------------------------------------------
    # E_ln_p_pi - E_ln_q_pi = -KL[q(pi) || p(pi)]
    # --------------------------------------------------
    dir_gamma = torch.distributions.dirichlet.Dirichlet(concentration=gamma)
    dirichlet_prior = torch.distributions.dirichlet.Dirichlet(concentration=alpha)
    KL_qpi_ppi = torch.distributions.kl._kl_dirichlet_dirichlet(p=dir_gamma, q=dirichlet_prior)

    return E_ln_pu + E_ln_pz - E_ln_qz - KL_qpi_ppi

def get_Newton_step(gamma: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Calculate the one step of Newton-Raphson method to update alpha

    Args:
        gamma: variational parameters of Dirichlet distribution
        alpha: the parameter of the Dirichlet distribution

    Return: Hinv_g
    """
    # first derivative
    g = dirichlet_expected_log(concentration=gamma) - dirichlet_expected_log(concentration=alpha)

    a = torch.polygamma(input=torch.sum(input=alpha, dim=0), n=1)

    q = - torch.polygamma(input=alpha, n=1)

    b = torch.sum(input=g / q, dim=0) / (1 / a + torch.sum(input=1 / q, dim=0))

    Hinv_g = (g - b) / q

    return Hinv_g

def make_scale_tril_matrix(scale_tril_log_diagonal: torch.Tensor, scale_tril_lower: torch.Tensor) -> torch.Tensor:
    """
    """
    K, nz = scale_tril_log_diagonal.shape

    scale_tril_diag = torch.exp(input=scale_tril_log_diagonal)
    scale_tril = torch.diag_embed(input=scale_tril_diag)
    tril_indices = torch.tril_indices(row=nz, col=nz, offset=-1)
    for k in range(K):
        scale_tril[k, tril_indices[0], tril_indices[1]] = scale_tril[k, tril_indices[0], tril_indices[1]] + scale_tril_lower[k]
    
    return scale_tril

def load_model(epoch_id: int = 0, lr: float = 1e-3) -> typing.Tuple[dict, torch.nn.Module, torch.nn.Module, torch.optim.Optimizer]:
    """Load or initialize model

    Args:
        epoch_id: which file to load
        lr: learning rate for VAE

    Returns:
        g_components: dictionary consisting of meta-components: mean and cov
        encoder: VAE network
        opt: optimizer for VAE
    """
    g_components = {
        'mean': 0.01 * torch.randn(size=(config['K'], config['nz']), device=device),
        'cov': torch.diag_embed(input=2 * torch.ones(size=(config['K'], config['nz']), device=device))
    }

    encoder = Encoder(nc=img_size[0], nef=config['nf'], nz=config['nz'], nzd=2, variational=True)
    encoder.apply(weights_init)
    encoder.to(device)

    decoder = Decoder(nc=img_size[0], ndf=config['nf'], nz=config['nz'])
    decoder.apply(weights_init)
    decoder.to(device)

    if epoch_id > 0:
        checkpoint_filename = 'Epoch_{0:d}.pt'.format(epoch_id)
        checkpoint_fullpath = os.path.join(config['logdir'], checkpoint_filename)
        if torch.cuda.is_available():
            saved_checkpoint = torch.load(
                checkpoint_fullpath,
                map_location=lambda storage,
                loc: storage.cuda(gpu_id)
            )
        else:
            saved_checkpoint = torch.load(
                checkpoint_fullpath,
                map_location=lambda storage,
                loc: storage
            )
        g_components = saved_checkpoint['g_components']

        encoder.load_state_dict(state_dict=saved_checkpoint['encoder'])
        decoder.load_state_dict(state_dict=saved_checkpoint['decoder'])

    # optimizer
    opt = torch.optim.Adam([
        {'params': encoder.parameters(), 'weight_decay': 0},
        {'params': decoder.parameters(), 'weight_decay': 0}
    ], lr=lr)

    return g_components, encoder, decoder, opt

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if config['train_flag']:
        train()