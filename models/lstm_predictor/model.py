import torch
import torch.nn as nn

class LSTMRiskPredictor(nn.Module):
    """
    LSTM model for equipment failure risk prediction

    Input  : Sequence of 10 BERT embeddings
             Shape: (batch, sequence_len, 768)
    Output : Risk probability (0 = safe, 1 = at risk)
    """

    def __init__(
        self,
        input_size  = 768,
        hidden_size = 128,
        num_layers  = 2,
        num_classes = 2,
        dropout     = 0.3
    ):
        super(LSTMRiskPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout,
            bidirectional = True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classification head
        # bidirectional so hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x shape: (batch, sequence_len, 768)
        """

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)

        # Take last time step output
        last_output = lstm_out[:, -1, :]

        # Apply dropout
        dropped = self.dropout(last_output)

        # Get predictions
        logits = self.classifier(dropped)

        return logits