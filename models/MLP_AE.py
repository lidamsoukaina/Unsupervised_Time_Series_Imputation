import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedAutoencoder(nn.Module):
    """Fully connected autoencoder"""

    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super().__init__()
        # Encoder: affine function
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        # Decoder: affine function
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_1)
        self.fc4 = nn.Linear(hidden_dim_1, output_dim)

    def forward(self, x):
        # Encoder
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)

        # Decoder
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = torch.sigmoid(out)

        return out
