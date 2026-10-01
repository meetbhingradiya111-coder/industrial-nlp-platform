import os
import sys
import json
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

from models.lstm_predictor.model   import LSTMRiskPredictor
from models.lstm_predictor.dataset import SequenceDataset

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)                                          # load hyperparameters

CONFIG = {
    # Data paths
    'train_path'   : 'data/sequences/train.csv',
    'val_path'     : 'data/sequences/val.csv',
    'save_path'    : 'models/lstm_predictor/',

    # Hyperparameters from params.yaml
    'sequence_len' : params['lstm_predictor']['sequence_len'],
    'hidden_size'  : params['lstm_predictor']['hidden_size'],
    'num_layers'   : params['lstm_predictor']['num_layers'],
    'dropout'      : params['lstm_predictor']['dropout'],
    'epochs'       : params['lstm_predictor']['epochs'],
    'batch_size'   : params['lstm_predictor']['batch_size'],
    'learning_rate': params['lstm_predictor']['learning_rate'],

    # Fixed settings
    'horizon'      : params['lstm_predictor'].get('horizon', 5),
    'num_classes'  : 2,
    'input_size'   : 768,
    'patience'     : 4,
    'seed'         : 42,
}


# ─────────────────────────────────────
# 2. REPRODUCIBILITY AND DEVICE
# ─────────────────────────────────────

random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
if torch.backends.mps.is_available():
    torch.mps.manual_seed(CONFIG['seed'])

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using M2 Mac GPU (MPS) ✅")
else:
    device = torch.device("cpu")
    print("Using CPU ✅")


# ─────────────────────────────────────
# 3. DATA LOADING
# ─────────────────────────────────────

def build_loaders():
    train_dataset = SequenceDataset(CONFIG['train_path'], CONFIG['sequence_len'], CONFIG['horizon'])
    val_dataset   = SequenceDataset(CONFIG['val_path'],   CONFIG['sequence_len'], CONFIG['horizon'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'], shuffle=False)

    print(f"\n✅ Train sequences: {len(train_dataset)} | Val sequences: {len(val_dataset)}")
    return train_dataset, train_loader, val_loader


def compute_class_weights(labels):
    num_normal  = labels.count(0)
    num_at_risk = labels.count(1)
    total       = len(labels)

    weights = [
        total / (2 * num_normal)  if num_normal  > 0 else 0.0,          # normal weight
        total / (2 * num_at_risk) if num_at_risk > 0 else 0.0,          # at-risk weight
    ]
    print(f"✅ Class weights -> Normal: {weights[0]:.3f} | At Risk: {weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ─────────────────────────────────────
# 4. TRAIN AND VALIDATE
# ─────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in loader:
        sequences = batch['sequence'].to(device)                        # move inputs
        labels    = batch['label'].to(device)                           # move labels

        optimizer.zero_grad()                                           # reset gradients
        loss = criterion(model(sequences), labels)                      # forward + loss
        loss.backward()                                                 # backward pass
        optimizer.step()                                                # update weights

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            sequences = batch['sequence'].to(device)
            labels    = batch['label'].to(device)

            outputs = model(sequences)
            total_loss += criterion(outputs, labels).item()

            probs = torch.softmax(outputs, dim=1)[:, 1]                 # at-risk probability
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / len(loader), np.array(all_probs), np.array(all_labels)


def score_at_threshold(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)                            # apply threshold
    return {
        'accuracy' : float((preds == labels).mean() * 100),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall'   : float(recall_score(labels, preds, zero_division=0)),
        'f1'       : float(f1_score(labels, preds, zero_division=0)),
    }


def find_best_threshold(probs, labels):
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.10, 0.91, 0.05):                       # sweep thresholds
        f1 = f1_score(labels, (probs >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)       # keep best F1
    return round(best_threshold, 2), best_f1


# ─────────────────────────────────────
# 5. MAIN TRAINING
# ─────────────────────────────────────

def train():
    print("\n" + "=" * 50)
    print("STARTING LSTM TRAINING")
    print("=" * 50)

    train_dataset, train_loader, val_loader = build_loaders()
    class_weights = compute_class_weights(train_dataset.labels)

    mlflow.set_experiment("LSTM_Risk_Predictor")

    with mlflow.start_run(run_name="LSTM_run_horizon"):
        mlflow.log_params(CONFIG)

        model = LSTMRiskPredictor(
            input_size  = CONFIG['input_size'],
            hidden_size = CONFIG['hidden_size'],
            num_layers  = CONFIG['num_layers'],
            num_classes = CONFIG['num_classes'],
            dropout     = CONFIG['dropout']
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        best_f1          = -1.0
        patience_counter = 0
        model_path       = os.path.join(CONFIG['save_path'], 'best_model.pt')

        for epoch in range(1, CONFIG['epochs'] + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, probs, labels = validate(model, val_loader, criterion)
            scheduler.step()

            scores = score_at_threshold(probs, labels, 0.5)             # metrics at default cutoff
            print(f"\nEpoch {epoch}/{CONFIG['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Val Acc: {scores['accuracy']:.2f}% | Precision: {scores['precision']:.3f} | "
                  f"Recall: {scores['recall']:.3f} | F1: {scores['f1']:.3f}")

            mlflow.log_metrics({
                'train_loss'    : train_loss,
                'val_loss'      : val_loss,
                'val_accuracy'  : scores['accuracy'],
                'val_precision' : scores['precision'],
                'val_recall'    : scores['recall'],
                'val_f1'        : scores['f1'],
            }, step=epoch)

            if scores['f1'] > best_f1:
                best_f1          = scores['f1']
                patience_counter = 0
                torch.save(model.state_dict(), model_path)              # save best by F1
                print(f"  ✅ Best model saved (F1: {best_f1:.3f})")
            else:
                patience_counter += 1
                print(f"  ⚠️  No improvement. Patience: {patience_counter}/{CONFIG['patience']}")

            if patience_counter >= CONFIG['patience']:
                print("\n⚠️  Early stopping triggered")
                break

        # Reload best checkpoint and pick the threshold on val
        model.load_state_dict(torch.load(model_path, map_location=device))
        _, probs, labels = validate(model, val_loader, criterion)
        threshold, tuned_f1 = find_best_threshold(probs, labels)
        final = score_at_threshold(probs, labels, threshold)
        pr_auc = float(average_precision_score(labels, probs))

        with open(os.path.join(CONFIG['save_path'], 'threshold.json'), 'w') as f:
            json.dump({'threshold': threshold, 'val_f1': tuned_f1,
                       'horizon': CONFIG['horizon']}, f, indent=2)      # save threshold

        print("\n" + "=" * 50)
        print("LSTM TRAINING COMPLETE")
        print("=" * 50)
        print(f"✅ Best threshold (val)  : {threshold}")
        print(f"✅ Val accuracy          : {final['accuracy']:.2f}%")
        print(f"✅ Val precision / recall: {final['precision']:.3f} / {final['recall']:.3f}")
        print(f"✅ Val F1                : {final['f1']:.3f}")
        print(f"✅ Val PR-AUC            : {pr_auc:.3f}")

        mlflow.log_metrics({
            'best_threshold'   : threshold,
            'tuned_val_f1'     : tuned_f1,
            'tuned_val_recall' : final['recall'],
            'val_pr_auc'       : pr_auc,
        })


if __name__ == "__main__":
    train()