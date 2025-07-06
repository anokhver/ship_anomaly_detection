import torch
from torch import nn

class LSTMModel(nn.Module):
    """
    A PyTorch implementation of a Long Short-Term Memory (LSTM) model for time-series forecasting.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 output_size,
                 dropout=0):

        super(LSTMModel, self).__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_size, input_size)

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Encode
        _, (hidden, cell) = self.encoder(x)

        # Prepare decoder input: repeat the encoded representation for each timestep
        # Use the last hidden state as the encoded representation
        encoded = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)  # Shape: (batch, seq_len, hidden_size)

        # Decode: Reconstruct the sequence
        decoder_output, _ = self.decoder(encoded, (hidden, cell))

        # Reconstruct
        reconstructed = self.output_layer(decoder_output)

        return reconstructed

    def get_reconstruction_error(self, x, loss):
        """Calculate reconstruction error for anomaly detection"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            # Calculate MSE for each sequence
            result =  loss(reconstructed, x).amax(dim=(1, 2)).cpu().numpy()
            return result

    def encode(self, x):
        """Get the encoded representation of input sequences"""
        with torch.no_grad():
            _, (hidden, _) = self.encoder(x)
            return hidden[-1]  # Return the last layer's hidden state
