import torch
import torch.nn as nn

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

CONFIG = {
    'input_dim'    : 768,   # BERT embedding size (was 300 for TF-IDF)
    'encoding_dim' : 64,    # Size of the compressed representation
    'dropout'      : 0.2,
}


# ─────────────────────────────────────
# 2. AUTOENCODER MODEL
# ─────────────────────────────────────

class LogAutoencoder(nn.Module):
    """
    Encoder-decoder network that learns to reconstruct normal maintenance logs
    from their BERT embeddings. High reconstruction error signals a possible anomaly.
    """

    def __init__(self, input_dim, encoding_dim, dropout=0.2):
        super().__init__()

        # Encoder: compresses input down to encoding_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),       # First compression step
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),             # Second compression step
            nn.ReLU(),
            nn.Linear(128, encoding_dim),    # Final compressed representation
            nn.ReLU()
        )

        # Decoder: reconstructs original input from compressed representation
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),    # First expansion step
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),             # Second expansion step
            nn.ReLU(),
            nn.Linear(256, input_dim)        # Back to original BERT embedding size
            # No Sigmoid here — BERT embeddings aren't bounded [0,1] like TF-IDF was
        )

    def forward(self, x):
        # Compress input into a smaller representation
        encoded = self.encoder(x)

        # Reconstruct the original input from that representation
        decoded = self.decoder(encoded)

        return decoded