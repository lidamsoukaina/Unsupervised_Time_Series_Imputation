import torch
import torch.nn as nn


class Encoder_LSTM(nn.Module):
    """Encoder LSTM"""

    def __init__(self, seq_len, no_features, embedding_size):
        super().__init__()
        self.seq_len = seq_len
        self.no_features = no_features
        self.embedding_size = embedding_size
        self.hidden_size = 2 * embedding_size
        self.LSTM1 = nn.LSTM(
            input_size=no_features,
            hidden_size=embedding_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        x, (hidden_state, cell_state) = self.LSTM1(x)
        last_lstm_layer_hidden_state = hidden_state[-1, :, :]
        return last_lstm_layer_hidden_state


class Decoder_LSTM(nn.Module):
    """Decoder LSTM"""

    def __init__(self, seq_len, no_features, output_size):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.hidden_size = 2 * no_features
        self.output_size = output_size
        self.LSTM1 = nn.LSTM(
            input_size=no_features,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))
        out = self.fc(x)
        return out


class LSTM_AE(nn.Module):
    """LSTM Autoencoder"""

    def __init__(self, seq_len, no_features, embedding_dim):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.embedding_dim = embedding_dim
        self.encoder = Encoder_LSTM(self.seq_len, self.no_features, self.embedding_dim)
        self.decoder = Decoder_LSTM(self.seq_len, self.embedding_dim, self.no_features)
        # Set requires_grad=True for LSTM parameters
        self.encoder.LSTM1.weight_ih_l0.requires_grad = True
        self.encoder.LSTM1.weight_hh_l0.requires_grad = True
        self.decoder.LSTM1.weight_ih_l0.requires_grad = True
        self.decoder.LSTM1.weight_hh_l0.requires_grad = True

    def forward(self, x):
        torch.manual_seed(0)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torch.sigmoid(decoded)
        return decoded

    def encode(self, x):
        self.eval()
        encoded = self.encoder(x)
        return encoded

    def decode(self, x):
        self.eval()
        decoded = self.decoder(x)
        squeezed_decoded = decoded.squeeze()
        return squeezed_decoded
