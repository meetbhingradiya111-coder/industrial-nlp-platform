import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import os

from models.autoencoder.model import LogAutoencoder
from models.autoencoder.dataset import LogAnomalyDataset

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

CONFIG = {
    # Data paths
    'train_path' : 'data/clean/train.csv',
    'val_path'   : 'data/clean/val.csv',
    'save_path'  : 'models/autoencoder/saved/',

    # Model settings
    'input_dim'    : 768,   # BERT embedding size
    'encoding_dim' : 64,
    'dropout'      : 0.2,

    # Training settings
    'epochs'        : 30,
    'batch_size'     : 32,
    'learning_rate'  : 0.001,
    'patience'       : 5,   # Early stopping patience
}


# ─────────────────────────────────────
# 2. DETECT DEVICE (M2 Mac = mps)
# ─────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using M2 Mac GPU (MPS) ✅")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA) ✅")
else:
    device = torch.device("cpu")
    print("Using CPU ✅")


# ─────────────────────────────────────
# 3. LOAD DATASETS
# ─────────────────────────────────────
# NOTE: This step embeds every log with BERT, so it takes a few
# minutes even on the M2 GPU. This runs once per training session.

print("\nLoading datasets (embedding with BERT, please wait)...")

train_dataset = LogAnomalyDataset(
    CONFIG['train_path'],
    filter_normal_only=True
)

val_dataset = LogAnomalyDataset(
    CONFIG['val_path'],
    filter_normal_only=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False
)

print(f"Train batches : {len(train_loader)}")
print(f"Val batches   : {len(val_loader)}")


# ─────────────────────────────────────
# 4. TRAINING FUNCTION
# ─────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train autoencoder for one complete epoch"""

    model.train()
    total_loss = 0

    for features, _ in loader:
        features = features.to(device)

        optimizer.zero_grad()

        # Forward pass: reconstruct the input embedding
        reconstructed = model(features)

        # Compare reconstruction to original embedding
        loss = criterion(reconstructed, features)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


# ─────────────────────────────────────
# 5. VALIDATION FUNCTION
# ─────────────────────────────────────

def validate(model, loader, criterion, device):
    """Evaluate reconstruction loss on validation set"""

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for features, _ in loader:
            features = features.to(device)
            reconstructed = model(features)
            loss = criterion(reconstructed, features)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


# ─────────────────────────────────────
# 6. MAIN TRAINING LOOP
# ─────────────────────────────────────

def train():
    print("\n" + "="*50)
    print("STARTING AUTOENCODER TRAINING (BERT EMBEDDINGS)")
    print("="*50)

    mlflow.set_experiment("Autoencoder_Anomaly_Detector_BERT")

    with mlflow.start_run(run_name="Autoencoder_BERT_run_1"):

        mlflow.log_params(CONFIG)

        model = LogAutoencoder(
            input_dim=CONFIG['input_dim'],
            encoding_dim=CONFIG['encoding_dim'],
            dropout=CONFIG['dropout']
        ).to(device)

        print(f"\nModel loaded to {device}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CONFIG['learning_rate']
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, CONFIG['epochs'] + 1):

            print(f"\n{'─'*40}")
            print(f"EPOCH {epoch}/{CONFIG['epochs']}")
            print(f"{'─'*40}")

            train_loss = train_one_epoch(
                model, train_loader,
                optimizer, criterion, device
            )

            val_loss = validate(
                model, val_loader,
                criterion, device
            )

            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss : {train_loss:.6f}")
            print(f"  Val Loss   : {val_loss:.6f}")

            mlflow.log_metrics({
                'train_loss' : train_loss,
                'val_loss'   : val_loss,
            }, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                os.makedirs(CONFIG['save_path'], exist_ok=True)
                torch.save(
                    model.state_dict(),
                    os.path.join(CONFIG['save_path'], 'best_model.pt')
                )
                print(f"\n  ✅ New best model saved!")
                print(f"     Val Loss: {val_loss:.6f}")

                mlflow.pytorch.log_model(model, "best_autoencoder_model")
            else:
                patience_counter += 1
                print(f"\n  No improvement. "
                      f"Patience: {patience_counter}/{CONFIG['patience']}")

            if patience_counter >= CONFIG['patience']:
                print(f"\nEarly stopping triggered!")
                print(f"Best Val Loss: {best_val_loss:.6f}")
                break

        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        print(f"Best Validation Loss: {best_val_loss:.6f}")

        mlflow.log_metric("best_val_loss", best_val_loss)

    return best_val_loss


# ─────────────────────────────────────
# 7. RUN TRAINING
# ─────────────────────────────────────

if __name__ == "__main__":
    best_loss = train()
    print(f"\nFinal Best Val Loss: {best_loss:.6f}")