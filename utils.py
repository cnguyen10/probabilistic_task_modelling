import random
import os
import typing as _typing
import numpy as np
import torch

def list_dir(root: str, prefix: bool = False) -> _typing.List[str]:
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories


def list_files(root: str, suffix: str, prefix: bool = False) -> _typing.List[str]:
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = [p for p in os.listdir(root) if os.path.isfile(os.path.join(root, p)) and p.endswith(suffix)]
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if m.weight is not None:
            torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)
    elif classname.find('BatchNorm') != -1:
        if m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)

def train_val_split(X: _typing.List[_typing.List[np.ndarray]], k_shot: int, shuffle: bool = True) -> _typing.Tuple[np.ndarray, _typing.List[int], np.ndarray, _typing.List[int]]:
    """Split data into train and validation

    Args:
        X: a list of sub-list of numpy array. 
            Each sub-list consists of data belonging to the same class
        k_shot: number of training data per class
        shuffle: shuffle data before splitting

    Returns:
    """
    # get information of image size
    nc, iH, iW = X[0][0].shape

    v_shot = len(X[0]) - k_shot
    num_classes = len(X)

    x_t = np.empty(shape=(num_classes, k_shot, nc, iH, iW))
    x_v = np.empty(shape=(num_classes, v_shot, nc, iH, iW))
    y_t = [0] * num_classes * k_shot
    y_v = [0] * num_classes * v_shot
    for cls_id in range(num_classes):
        if shuffle:
            random.shuffle(x=X[cls_id]) # in-place shuffle data within the same class
        x_t[cls_id, :, :, :, :] = np.array(X[cls_id][:k_shot])
        x_v[cls_id, :, :, :, :] = np.array(X[cls_id][k_shot:])
        y_t[k_shot * cls_id: k_shot * (cls_id + 1)] = [cls_id] * k_shot
        y_v[v_shot * cls_id: v_shot * (cls_id + 1)] = [cls_id] * v_shot

    x_t = np.concatenate(x_t, axis=0) # (C * k_shot, nc, iH, iW)
    x_v = np.concatenate(x_v, axis=0)

    return x_t, y_t, x_v, y_v

def euclidean_distance(matrixN: torch.Tensor, matrixM: torch.Tensor) -> torch.Tensor:
    """Calculate Euclidean distance from N points to M points

    Args:
        matrixN: an N x D matrix for N points
        matrixM: a M x D matrix for M points

    Returns: N x M matrix
    """
    N = matrixN.size(0)
    M = matrixM.size(0)
    D = matrixN.size(1)
    assert D == matrixM.size(1)

    matrixN = matrixN.unsqueeze(1).expand(N, M, D)
    matrixM = matrixM.unsqueeze(0).expand(N, M, D)

    return torch.norm(input=matrixN - matrixM, p='fro', dim=2)

def get_cls_prototypes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate the prototypes/centroids

    Args:
        x: input data
        y: corresponding labels

    Returns: a tensor of prototypes with shape (C, d),
        where C is the number of classes, d is the embedding dimension
    """
    _, d = x.shape
    cls_idx = torch.unique(input=y, return_counts=False)
    C = cls_idx.shape[0]

    prototypes = torch.empty(size=(C, d), device=x.device)
    for c in range(C):
        prototypes[c, :] = torch.mean(input=x[y == cls_idx[c]], dim=0)

    return prototypes


def get_probabilistic_prototypes(Nx: dict) -> dict:
    """Calculate the probabilistic prototypes of iid random variable x_i ~ N_i
    Since c = mean(x), then c ~ N(x; mean(mean_x), mean(cov_x))
    
    Args:
        Nx: dictionary with keys: mean, cov, y

    Return: prototypes in a dictionary with keys: mean, cov
    """
    _, d = Nx['mean'].shape
    cls_idx = torch.unique(input=Nx['y'], return_counts=False)
    C = cls_idx.shape[0]

    prototypes = {
        'mean': torch.empty(size=(C, d), device=Nx['mean'].device),
        'cov': torch.empty(size=(C, d, d), device=Nx['cov'].device)
    }

    for c in range(C):
        idx = Nx['y'] == cls_idx[c] # bool
        num_x = torch.sum(input=idx, dim=0)
        prototypes['mean'][c, :] = torch.sum(input=Nx['mean'][idx], dim=0) / num_x
        prototypes['cov'][c, :, :] = torch.sum(input=Nx['cov'][idx], dim=0) / torch.square(input=num_x)
    
    return prototypes

def log_int_prod_gaussians(N1: dict, N2: dict) -> torch.Tensor:
    """Calculate log integral N(x; m1, cov1) * N(x; m2, cov2) dx

    Args:
        N = {'mean': torch.Tensor(k, d), 'cov': torch.Tensor(k, d, d)}

    Returns: k1-by-k2 matrix
    """
    # get number of components
    k1, d = N1['mean'].shape
    k2, d2 = N2['mean'].shape

    assert d == d2

    # expand size
    m1 = torch.unsqueeze(input=N1['mean'], dim=1).expand(k1, k2, d)
    cov1 = torch.unsqueeze(input=N1['cov'], dim=1).expand(k1, k2, d, d)

    m2 = torch.unsqueeze(input=N2['mean'], dim=0).expand(k1, k2, d)
    cov2 = torch.unsqueeze(input=N2['cov'], dim=0).expand(k1, k2, d, d)

    mvn = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=m1 - m2,
        covariance_matrix=cov1 + cov2
    )

    return mvn.log_prob(value=torch.zeros_like(N2['mean'], device=N2['mean'].device))

# def squared_L2_distance_gaussians(N1: dict, N2: dict) -> torch.Tensor:
#     """calculate the L2 square distance between 2 Gaussians
#     int (N1 - N2)**2 dx = int N1 * N1 dx - 2 int N1 * N2 dx + int N2 * N2 dx
#     Note that: log int N * N dx = -k / 2 * log(2 * pi) - 1 / 2 log det (2 * cov)

#     Args:
#         N: dictionary with 'mean' and 'cov'
    
#     Return: a n1-by-n2 distance matrix
#     """
#     n1, d = N1['mean'].shape
#     n2, d2 = N2['mean'].shape

#     assert d == d2

#     int_N1_N1 = torch.exp(
#         input=torch.diagonal(input=log_int_prod_gaussians(N1=N1, N2=N1), dim1=-2, dim2=-1)
#     )
#     int_N2_N2 = torch.exp(
#         input=torch.diagonal(input=log_int_prod_gaussians(N1=N2, N2=N2), dim1=-2, dim2=-1)
#     )

#     # expand shapes
#     int_N11 = torch.unsqueeze(input=int_N1_N1, dim=1).expand(n1, n2)
#     int_N22 = torch.unsqueeze(input=int_N2_N2, dim=0).expand(n1, n2)

#     int_N12 = torch.exp(input=log_int_prod_gaussians(N1=N1, N2=N2))

#     L2_square = int_N11 - 2 * int_N12 + int_N22

#     # normalize for numerical stability
#     L2_square = L2_square - torch.min(input=L2_square)

#     return L2_square

def squared_L2_distance_gaussians(N1: dict, N2: dict) -> torch.Tensor:
    """calculate the L2 square distance between 2 Gaussians
    int (N1 - N2)**2 dx = int N1 * N1 dx - 2 int N1 * N2 dx + int N2 * N2 dx
    Note that: log int N * N dx = -k / 2 * log(2 * pi) - 1 / 2 log det (2 * cov)

    Args:
        N: dictionary with 'mean' and 'cov'
    
    Return: a n1-by-n2 distance matrix
    """
    n1, d = N1['mean'].shape
    n2, d2 = N2['mean'].shape

    assert d == d2

    log_int_N1_N1 = torch.diagonal(input=log_int_prod_gaussians(N1=N1, N2=N1), dim1=-2, dim2=-1)
    log_int_N2_N2 = torch.diagonal(input=log_int_prod_gaussians(N1=N2, N2=N2), dim1=-2, dim2=-1)

    # expand shapes
    log_int_N11 = torch.unsqueeze(input=log_int_N1_N1, dim=1).expand(n1, n2)
    log_int_N22 = torch.unsqueeze(input=log_int_N2_N2, dim=0).expand(n1, n2)

    log_int_N12 = log_int_prod_gaussians(N1=N1, N2=N2)

    log_L2_square = torch.logaddexp(input=log_int_N11, other=log_int_N22)
    log_L2_square = logsubexp(x=log_L2_square, y=log_int_N12)
    log_L2_square = logsubexp(x=log_L2_square, y=log_int_N12)

    # return torch.exp(input=log_L2_square / 2)
    return log_L2_square / 2

def logsubexp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate log(exp(x) - exp(y))
    """
    return x + torch.log1p(input=-torch.exp(y - x))

def dict_to_torchGMM(gmm: dict) -> torch.distributions.MixtureSameFamily:
    """Convert a dictionary to a Gaussian mixture distribution in PyTorch

    Args:
        gmm: dictionary of GMM parameters
    
    Return: a GMM in PyTorch distribution
    """
    mixture_distribution = torch.distributions.Categorical(probs=gmm['pi'])
    comp_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=gmm['mean'],
        covariance_matrix=gmm['cov']
    )
    gm_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
        mixture_distribution=mixture_distribution,
        component_distribution=comp_distribution
    )
    return gm_distribution

def KL_div_GMM(gmm1: dict, gmm2: dict, num_samples: int) -> torch.Tensor:
    """Calculate KL[gmm1 || gmm2] by sampling

    Args:
        gmm1, gmm2: dictionaries containing 'mean', 'cov', and 'pi'
        num_samples: number of samples used to evaluate the KL divergence

    Return: KL divergence
    """
    gm1 = dict_to_torchGMM(gmm=gmm1)
    gm2 = dict_to_torchGMM(gmm=gmm2)

    samples = gm1.sample(sample_shape=(num_samples,))

    kl_div = gm1.log_prob(x=samples) - gm2.log_prob(x=samples)

    return torch.mean(input=kl_div)

def gaussian_log_prob(x: torch.Tensor, gm: dict) -> torch.Tensor:
    """calculate the Gaussian log-likelihood of data x

    Args:
        x: tensor of size (N, 1, d)
        gm: mean and cov of sizes (K, d) and (K, d, d)
    
    Return: N-by-K matrix of log-likelihood
    """
    if x.dim() == 2:
        x = torch.unsqueeze(input=x, dim=1)

    if 'scale_tril' in gm.keys():
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=gm['mean'],
            scale_tril=gm['scale_tril']
        )
    elif 'cov' in gm.keys():
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=gm['mean'],
            covariance_matrix=gm['cov']
        )
    else:
        raise ValueError('Missing scale_tril or cov')

    return mvn.log_prob(value=x) # (N, K)

def gaussian_mixture_log_prob(x: torch.Tensor, gmm: dict) -> torch.Tensor:
    """Calculate the log-likelihood of a Gaussian mixture model

    Args:
        x: data to evaluate the weighted log-likelihood (N, d)
        gmm: dictionary of a Gaussian mixture model (K components)

    Return: a N-dimensional log-likelihood vector
    """
    mixture_distribution = torch.distributions.Categorical(probs=gmm['pi'])

    if 'scale_tril' in gmm.keys():
        comp_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=gmm['mean'],
            scale_tril=gmm['scale_tril']
        )
    elif 'cov' in gmm.keys():
        comp_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=gmm['mean'],
            covariance_matrix=gmm['cov']
        )
    else:
        raise ValueError('Missing scale_tril or cov')

    gm_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
        mixture_distribution=mixture_distribution,
        component_distribution=comp_distribution
    )

    return gm_distribution.log_prob(x=x)

def dirichlet_expected_log(concentration: torch.Tensor) -> torch.Tensor:
    """
    """
    return torch.digamma(input=concentration) - torch.digamma(input=torch.sum(input=concentration, dim=-1))

def get_reconstruction_loss(x: torch.Tensor, z: torch.Tensor, decoder: torch.nn.Module) -> torch.Tensor:
    """
    Calculate the reconstruction of the VAE
    Args:
        - x: original data
        - m, s: mean and diagonal std of encoding data
        - decoder
    """
    x_reconstruct = decoder.forward(z=z) # (N, C, H, W)
    c_bernoulli = torch.distributions.continuous_bernoulli.ContinuousBernoulli(logits=x_reconstruct)
    reconstruction_loss = - c_bernoulli.log_prob(value=x) # (N, C, H, W)

    return torch.mean(input=reconstruction_loss)

def get_entropy_diagonal_cov(s: torch.Tensor) -> torch.Tensor:
    """Calculate entropy of a diagonal Gaussian distribution

    Args:
        s: vector (or a batch of vector) of standard deviation (size = (*, nz))
    """
    # get dimension
    nz = s.shape[-1]

    entropy = nz / 2 * (np.log(2 * np.pi) + 1)
    entropy = entropy + torch.sum(input=torch.log(input=s), dim=-1)

    return torch.mean(input=entropy, dim=0)

def wishart_log_prob(x: torch.Tensor, W: torch.Tensor, df: float) -> torch.Tensor:
    """
    """
    p = W.shape[0]

    assert df > (p - 1)
    
    log_prob = (df - p - 1) / 2 * torch.logdet(input=x)
    log_prob = log_prob - batch_trace(x=torch.matmul(input=torch.inverse(input=W), other=x)) / 2
    log_prob = log_prob - df * p / 2 * np.log(2)
    log_prob = log_prob - df / 2 * torch.logdet(input=W)
    log_prob = log_prob - torch.mvlgamma(input=torch.tensor(df / 2, dtype=torch.float, device=x.device), p=p)

    return log_prob

def inverse_wishart_log_prob(x: torch.Tensor, W: torch.Tensor, df: float) -> torch.float:
    """
    """
    p = W.shape[0]

    assert df > (p - 1)

    log_prob = -(df + p + 1) / 2 * torch.logdet(input=x)
    log_prob = log_prob - batch_trace(x=torch.matmul(input=W, other=torch.inverse(input=x))) / 2
    log_prob = log_prob + df / 2 * torch.logdet(input=W)
    log_prob = log_prob - df * p / 2 * np.log(2)
    log_prob = log_prob - torch.mvlgamma(input=torch.tensor(df / 2, dtype=torch.float, device=x.device), p=p)

    return log_prob

def batch_trace(x: torch.Tensor) -> torch.Tensor:
    """
    """
    return torch.sum(input=torch.diagonal(input=x, dim1=-2, dim2=-1), dim=-1)