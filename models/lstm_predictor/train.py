import torch
import torch.nn as nn
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import yaml
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

from models.lstm_predictor.model   import LSTMRiskPredictor
from models.lstm_predictor.dataset import SequenceDataset

# ─────────────────────────────────────
# LOAD HYPERPARAMETERS FROM params.yaml
# ─────────────────────────────────────
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

CONFIG = {
    # Data paths
    'train_path'   : 'data/clean/train.csv',
    'val_path'     : 'data/clean/val.csv',
    'save_path'    : 'models/lstm_predictor/',

    # Hyperparameters from params.yaml
    'sequence_len' : params['lstm_predictor']['sequence_len'],
    'hidden_size'  : params['lstm_predictor']['hidden_size'],
    'num_layers'   : params['lstm_predictor']['num_layers'],
    'dropout'      : params['lstm_predictor']['dropout'],
    'epochs'       : params['lstm_predictor']['epochs'],
    'batch_size'   : params['lstm_predictor']['batch_size'],
    'learning_rate': params['lstm_predictor']['learning_rate'],
    'num_classes'  : 2,
    'input_size'   : 768,
    'patience'     : 3,
}

print("\nHyperparameters loaded from params.yaml:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# ─────────────────────────────────────
# DETECT DEVICE
# ─────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\nUsing M2 Mac GPU (MPS) ✅")
else:
    device = torch.device("cpu")
    print("\nUsing CPU ✅")

# ─────────────────────────────────────
# LOAD DATASETS
# ─────────────────────────────────────
print("\nCreating sequence datasets...")
print("This may take 5-10 minutes...")
print("BERT is generating embeddings for all logs...")

train_dataset = SequenceDataset(
    CONFIG['train_path'],
    CONFIG['sequence_len']
)
val_dataset = SequenceDataset(
    CONFIG['val_path'],
    CONFIG['sequence_len']
)

train_loader = DataLoader(
    train_dataset,
    batch_size = CONFIG['batch_size'],
    shuffle    = True
)
val_loader = DataLoader(
    val_dataset,
    batch_size = CONFIG['batch_size'],
    shuffle    = False
)

print(f"\nTrain sequences : {len(train_dataset)}")
print(f"Val sequences   : {len(val_dataset)}")
print(f"Train batches   : {len(train_loader)}")
print(f"Val batches     : {len(val_loader)}")

# ─────────────────────────────────────
# COMPUTE CLASS WEIGHTS (address imbalance)
# ─────────────────────────────────────
# "At risk" sequences are the minority class. Without weighting,
# the model can minimize loss by always predicting "normal" -
# weighting penalizes that shortcut more heavily.

train_labels = train_dataset.labels
num_normal = train_labels.count(0)
num_at_risk = train_labels.count(1)
total = len(train_labels)

print(f"\nTrain label distribution:")
print(f"  Normal  : {num_normal}")
print(f"  At Risk : {num_at_risk}")

class_weights = [
    total / (2 * num_normal) if num_normal > 0 else 0.0,
    total / (2 * num_at_risk) if num_at_risk > 0 else 0.0,
]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print(f"\nComputed class weights:")
print(f"  Normal  : {class_weights[0]:.3f}")
print(f"  At Risk : {class_weights[1]:.3f}")

# ─────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for batch_idx, batch in enumerate(loader):
        sequences = batch['sequence'].to(device)
        labels    = batch['label'].to(device)

        optimizer.zero_grad()
        outputs     = model(sequences)
        loss        = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        correct    += (predictions == labels).sum().item()
        total      += labels.size(0)
        total_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} "
                  f"Loss: {loss.item():.4f}")

    return total_loss / len(loader), correct / total * 100

# ─────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for batch in loader:
            sequences = batch['sequence'].to(device)
            labels    = batch['label'].to(device)

            outputs     = model(sequences)
            loss        = criterion(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)

            correct    += (predictions == labels).sum().item()
            total      += labels.size(0)
            total_loss += loss.item()

    return total_loss / len(loader), correct / total * 100

# ─────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────
def train():
    print("\n" + "="*50)
    print("STARTING LSTM TRAINING")
    print("="*50)

    mlflow.set_experiment("LSTM_Risk_Predictor")

    with mlflow.start_run(run_name="LSTM_run_1"):
        mlflow.log_params(CONFIG)

        model = LSTMRiskPredictor(
            input_size  = CONFIG['input_size'],
            hidden_size = CONFIG['hidden_size'],
            num_layers  = CONFIG['num_layers'],
            num_classes = CONFIG['num_classes'],
            dropout     = CONFIG['dropout']
        ).to(device)

        print(f"\nModel loaded to: {device}")

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = CONFIG['learning_rate']
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )

        best_val_accuracy = 0
        patience_counter  = 0

        for epoch in range(1, CONFIG['epochs'] + 1):

            print(f"\n{'─'*40}")
            print(f"EPOCH {epoch}/{CONFIG['epochs']}")
            print(f"{'─'*40}")

            train_loss, train_acc = train_one_epoch(
                model, train_loader,
                optimizer, criterion
            )
            val_loss, val_acc = validate(
                model, val_loader, criterion
            )
            scheduler.step()

            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Accuracy : {train_acc:.2f}%")
            print(f"  Val Accuracy   : {val_acc:.2f}%")
            print(f"  Train Loss     : {train_loss:.4f}")
            print(f"  Val Loss       : {val_loss:.4f}")

            mlflow.log_metrics({
                'train_loss'     : train_loss,
                'train_accuracy' : train_acc,
                'val_loss'       : val_loss,
                'val_accuracy'   : val_acc,
            }, step=epoch)

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter  = 0
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        CONFIG['save_path'],
                        'best_model.pt'
                    )
                )
                mlflow.pytorch.log_model(
                    model, "best_lstm_model"
                )
                print(f"  ✅ Best model saved! "
                      f"Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"  No improvement. "
                      f"Patience: {patience_counter}"
                      f"/{CONFIG['patience']}")

            if patience_counter >= CONFIG['patience']:
                print("\nEarly stopping triggered!")
                break

        print("\n" + "="*50)
        print("LSTM TRAINING COMPLETE")
        print("="*50)
        print(f"Best Val Accuracy: {best_val_accuracy:.2f}%")

        if best_val_accuracy >= 80:
            print("✅ TARGET ACHIEVED!")
        else:
            print("⚠️  Below 80% target.")

        mlflow.log_metric(
            "best_val_accuracy",
            best_val_accuracy
        )

    return best_val_accuracy

if __name__ == "__main__":
    train()