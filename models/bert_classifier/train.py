# models/bert_classifier/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
import mlflow
import mlflow.pytorch
import json
import os
import sys

sys.path.append('../..')

from models.bert_classifier.model   import BERTFailureClassifier
from models.bert_classifier.dataset import MaintenanceLogDataset

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

CONFIG = {
    # Data paths
    'train_path'    : 'data/clean/train.csv',
    'val_path'      : 'data/clean/val.csv',
    'mapping_path'  : 'data/clean/failure_mapping.json',
    'save_path'     : 'models/bert_classifier/',

    # Model settings
    'num_classes'   : 6,
    'max_length'    : 128,
    'dropout'       : 0.3,

    # Training settings
    'epochs'        : 10,
    'batch_size'    : 16,
    'learning_rate' : 2e-5,
    'patience'      : 3,    # Early stopping patience
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
# 3. LOAD TOKENIZER
# ─────────────────────────────────────

print("\nLoading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained(
    'distilbert-base-uncased'
)
print("Tokenizer loaded ✅")

# ─────────────────────────────────────
# 4. LOAD DATASETS
# ─────────────────────────────────────

print("\nLoading datasets...")
train_dataset = MaintenanceLogDataset(
    CONFIG['train_path'],
    tokenizer,
    CONFIG['max_length']
)
val_dataset = MaintenanceLogDataset(
    CONFIG['val_path'],
    tokenizer,
    CONFIG['max_length']
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

print(f"Train batches : {len(train_loader)}")
print(f"Val batches   : {len(val_loader)}")

# ─────────────────────────────────────
# 5. LOAD LABEL MAPPING
# ─────────────────────────────────────

with open(CONFIG['mapping_path'], 'r') as f:
    failure_mapping = json.load(f)

print("\nFailure Categories:")
for k, v in failure_mapping.items():
    print(f"  {k} → {v}")

# ─────────────────────────────────────
# 5b. COMPUTE CLASS WEIGHTS (for imbalanced data)
# ─────────────────────────────────────
# No Failure (~50%) heavily outweighs fault classes (~8% each).
# Class weights make the loss function penalize mistakes on rare
# classes more heavily, so the model doesn't just learn to favor
# the majority class.

import pandas as pd
import numpy as np

train_df = pd.read_csv(CONFIG['train_path'])
class_counts = train_df['failure_label'].value_counts().sort_index()

print("\nClass distribution (train set):")
for label_id, count in class_counts.items():
    print(f"  {failure_mapping[str(label_id)]:22s}: {count}")

# Inverse-frequency weighting: rarer classes get higher weight
total_samples = len(train_df)
num_classes = CONFIG['num_classes']
class_weights = total_samples / (num_classes * class_counts.values)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("\nComputed class weights:")
for label_id, weight in zip(class_counts.index, class_weights):
    print(f"  {failure_mapping[str(label_id)]:22s}: {weight:.3f}")

# ─────────────────────────────────────
# 6. TRAINING FUNCTION
# ─────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train model for one complete epoch"""

    model.train()
    total_loss     = 0
    correct        = 0
    total          = 0

    for batch_idx, batch in enumerate(loader):

        # Move data to device
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Track accuracy
        predictions = torch.argmax(outputs, dim=1)
        correct    += (predictions == labels).sum().item()
        total      += labels.size(0)
        total_loss += loss.item()

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100

    return avg_loss, accuracy

# ─────────────────────────────────────
# 7. VALIDATION FUNCTION
# ─────────────────────────────────────

def validate(model, loader, criterion, device):
    """Evaluate model on validation set"""

    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for batch in loader:

            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            outputs     = model(input_ids, attention_mask)
            loss        = criterion(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)

            correct    += (predictions == labels).sum().item()
            total      += labels.size(0)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100

    return avg_loss, accuracy

# ─────────────────────────────────────
# 8. MAIN TRAINING LOOP
# ─────────────────────────────────────

def train():
    print("\n" + "="*50)
    print("STARTING BERT TRAINING")
    print("="*50)

    # Start MLflow experiment
    mlflow.set_experiment("BERT_Failure_Classifier")

    with mlflow.start_run(run_name="BERT_run_1"):

        # Log config to MLflow
        mlflow.log_params(CONFIG)

        # Initialize model
        model = BERTFailureClassifier(
            num_classes = CONFIG['num_classes'],
            dropout     = CONFIG['dropout']
        ).to(device)

        print(f"\nModel loaded to {device}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        # Loss function and optimizer
        # Class-weighted to address imbalance (No Failure ~50% vs fault classes ~8% each)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = CONFIG['learning_rate'],
            weight_decay = 0.01
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = 3,
            gamma     = 0.5
        )

        # Training variables
        best_val_accuracy = 0
        patience_counter  = 0
        history           = []

        # ── EPOCH LOOP ──────────────────────
        for epoch in range(1, CONFIG['epochs'] + 1):

            print(f"\n{'─'*40}")
            print(f"EPOCH {epoch}/{CONFIG['epochs']}")
            print(f"{'─'*40}")

            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader,
                optimizer, criterion, device
            )

            # Validate
            val_loss, val_acc = validate(
                model, val_loader,
                criterion, device
            )

            # Update learning rate
            scheduler.step()

            # Print results
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss     : {train_loss:.4f}")
            print(f"  Train Accuracy : {train_acc:.2f}%")
            print(f"  Val Loss       : {val_loss:.4f}")
            print(f"  Val Accuracy   : {val_acc:.2f}%")

            # Log to MLflow
            mlflow.log_metrics({
                'train_loss'     : train_loss,
                'train_accuracy' : train_acc,
                'val_loss'       : val_loss,
                'val_accuracy'   : val_acc,
            }, step=epoch)

            # Save history
            history.append({
                'epoch'         : epoch,
                'train_loss'    : train_loss,
                'train_accuracy': train_acc,
                'val_loss'      : val_loss,
                'val_accuracy'  : val_acc,
            })

            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter  = 0

                # Save model
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        CONFIG['save_path'],
                        'best_model.pt'
                    )
                )
                print(f"\n  ✅ New best model saved!")
                print(f"     Val Accuracy: {val_acc:.2f}%")

                # Log to MLflow
                mlflow.pytorch.log_model(
                    model,
                    "best_bert_model"
                )
            else:
                patience_counter += 1
                print(f"\n  No improvement. "
                      f"Patience: {patience_counter}"
                      f"/{CONFIG['patience']}")

            # Early stopping
            if patience_counter >= CONFIG['patience']:
                print(f"\nEarly stopping triggered!")
                print(f"Best Val Accuracy: {best_val_accuracy:.2f}%")
                break

        # ── TRAINING COMPLETE ────────────────
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")

        # Log final metrics
        mlflow.log_metric("best_val_accuracy", best_val_accuracy)

        # Target check
        if best_val_accuracy >= 85:
            print("✅ TARGET ACHIEVED! Accuracy >= 85%")
        else:
            print("⚠️  Below target. Consider tuning.")

    return best_val_accuracy, history

# ─────────────────────────────────────
# 9. RUN TRAINING
# ─────────────────────────────────────

if __name__ == "__main__":
    best_acc, history = train()
    print(f"\nFinal Best Accuracy: {best_acc:.2f}%")