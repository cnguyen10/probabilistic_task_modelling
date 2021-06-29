import torch

def initialize_decoder_module(in_channels: int, out_channels: int) -> torch.nn.Module:
    """Initialize a decoder module including a ConvT, batchnorm and ReLU
    """
    m = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        torch.nn.BatchNorm2d(num_features=out_channels, momentum=1, track_running_stats=False),
        torch.nn.LeakyReLU(negative_slope=1e-2)
    )
    return m

class Encoder(torch.nn.Module):
    def __init__(self, nc: int, nef: int, nz: int, nzd: int = 2, variational: bool = False) -> None:
        """Initialize an instance of the encoder

        Args:
            nc: number of input channels (1 for gray-scale images and 3 for color images)
            nef: based number of the channels (nef --> 2 * nef --> 4 * nef --> 8 * nef --> 16 * nef)
            nz: the dimension of the latent variable
            nzd: the dimension of the latent variable right after the convolutional layer and before the flattening layer
        """
        super(Encoder, self).__init__()
        self.nz = nz
        self.nc = nc
        self.nef = nef

        self.nzd = nzd

        self.variational = variational

        self.encoding = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.nc, out_channels=self.nef, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=self.nef, momentum=1, track_running_stats=False),
            torch.nn.LeakyReLU(negative_slope=1e-2),
            #
            torch.nn.Conv2d(in_channels=self.nef, out_channels=self.nef * 2, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=self.nef * 2, momentum=1, track_running_stats=False),
            torch.nn.LeakyReLU(negative_slope=1e-2),
            #
            torch.nn.Conv2d(in_channels=self.nef * 2, out_channels=self.nef * 4, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=self.nef * 4, momentum=1, track_running_stats=False),
            torch.nn.LeakyReLU(negative_slope=1e-2),
            #
            torch.nn.Conv2d(in_channels=self.nef * 4, out_channels=self.nef * 8, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=self.nef * 8, momentum=1, track_running_stats=False),
            torch.nn.LeakyReLU(negative_slope=1e-2),
            #
            torch.nn.Conv2d(in_channels=self.nef * 8, out_channels=self.nef * 16, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=self.nef * 16, momentum=1, track_running_stats=False),
            torch.nn.LeakyReLU(negative_slope=1e-2),
            
            torch.nn.Flatten(start_dim=1, end_dim=-1)
        )

        if not self.variational:
            self.encoding2 = torch.nn.Linear(in_features=nef * 16 * self.nzd**2, out_features=nz)
        else:
            self.encoding2_m = torch.nn.Linear(in_features=nef * 16 * self.nzd**2, out_features=nz)
            self.encoding2_s = torch.nn.Linear(in_features=nef * 16 * self.nzd**2, out_features=nz)
    
    def forward(self, x) -> torch.Tensor:
        h = self.encoding(x)
        if not self.variational:
            return self.encoding2(h)
        else:
            m = self.encoding2_m(h)
            log_s = self.encoding2_s(h)
            s = torch.exp(input=log_s)
            return m, s

class Decoder(torch.nn.Module):
    def __init__(self, nc: int, ndf: int, nz: int, nzd: int = 2) -> None:
        """Initialize an instance of the decoder

        Args:
            nc: number of input channels (1 for gray-scale images and 3 for color images)
            nef: based number of the channels (16 * nef --> 8 * nef --> 4 * nef --> 2 * nef --> nef --> nc)
            nz: the dimension of the latent variable
            nzd: see the description of the encoder
        """
        super(Decoder, self).__init__()
        self.nz = nz
        self.nc = nc
        self.ndf = ndf

        self.nzd = nzd

        self.decode1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=nz, out_features=ndf * 16 * self.nzd**2),
            torch.nn.LeakyReLU(negative_slope=1e-2)
        )

        self.decode2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=ndf * 16, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=ndf * 8, momentum=1, track_running_stats=False),
            torch.nn.LeakyReLU(negative_slope=1e-2),
            #
            torch.nn.ConvTranspose2d(in_channels=ndf * 8, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=ndf * 4, momentum=1, track_running_stats=False),
            torch.nn.LeakyReLU(negative_slope=1e-2),
            #
            torch.nn.ConvTranspose2d(in_channels=ndf * 4, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=ndf * 2, momentum=1, track_running_stats=False),
            torch.nn.LeakyReLU(negative_slope=1e-2),
            #
            torch.nn.ConvTranspose2d(in_channels=ndf * 2, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=ndf, momentum=1, track_running_stats=False),
            torch.nn.LeakyReLU(negative_slope=1e-2),
            #
            torch.nn.ConvTranspose2d(in_channels=ndf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False)
        )

    def forward(self, z) -> torch.Tensor:
        h = self.decode1(z)
        h_reshape = h.view(-1, self.ndf * 16, self.nzd, self.nzd)
        out = self.decode2(h_reshape)
        return out