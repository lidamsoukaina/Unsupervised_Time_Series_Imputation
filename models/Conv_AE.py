import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder"""

    def __init__(
        self,
        T,
        D,
        hidden_dim1,
        hidden_dim2,
        hidden_dim3,
        in_channels=1,
        out_channels_1=16,
        out_channels_2=8,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super(ConvAutoencoder, self).__init__()
        self.T = T
        self.D = D
        ## Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels_1,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels_1,
                out_channels=out_channels_2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(out_channels_2 * T * D, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
        )

        ## Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim3, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, out_channels_2 * T * D),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(out_channels_2, T, D)),
            nn.ConvTranspose2d(
                in_channels=out_channels_2,
                out_channels=out_channels_1,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=out_channels_1,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(
            1
        )  # add a channel dimension to the input (N, T, D) -> (N, 1, T, D)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze(
            1
        )  # remove the channel dimension from the output (N, 1, T, D) -> (N, T, D)
        return x
