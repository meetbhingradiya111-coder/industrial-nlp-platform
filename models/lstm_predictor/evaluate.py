import os
import json
import numpy as np
import torch
import mlflow
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, average_precision_score
)

from models.lstm_predictor.model import LSTMRiskPredictor
from models.lstm_predictor.dataset import SequenceDataset

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

CONFIG = {
    'test_path'      : 'data/sequences/test.csv',
    'model_path'     : 'models/lstm_predictor/best_model.pt',
    'threshold_path' : 'models/lstm_predictor/threshold.json',
    'sequence_len'   : 10,
    'hidden_size'    : 128,
    'num_layers'     : 2,
    'dropout'        : 0.3,
    'batch_size'     : 32,
}


# ─────────────────────────────────────
# 2. DEVICE
# ─────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device} ✅")


# ─────────────────────────────────────
# 3. LOAD SAVED THRESHOLD
# ─────────────────────────────────────

with open(CONFIG['threshold_path'], 'r') as f:
    saved = json.load(f)                                                # threshold picked on val

THRESHOLD = saved['threshold']
HORIZON   = saved['horizon']
print(f"✅ Threshold from val: {THRESHOLD} | Horizon: {HORIZON}")


# ─────────────────────────────────────
# 4. LOAD TEST DATA AND MODEL
# ─────────────────────────────────────

print("\nBuilding test sequences...")
test_dataset = SequenceDataset(CONFIG['test_path'], CONFIG['sequence_len'], HORIZON)
test_loader  = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

model = LSTMRiskPredictor(
    input_size  = 768,
    hidden_size = CONFIG['hidden_size'],
    num_layers  = CONFIG['num_layers'],
    num_classes = 2,
    dropout     = CONFIG['dropout']
).to(device)
model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
model.eval()
print("✅ Model loaded")


# ─────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────

def get_probabilities():
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['sequence'].to(device))               # forward pass
            probs   = torch.softmax(outputs, dim=1)[:, 1]               # at-risk probability
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(batch['label'].tolist())

    return np.array(all_probs), np.array(all_labels)


def score(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)                            # apply threshold
    return {
        'accuracy' : float((preds == labels).mean() * 100),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall'   : float(recall_score(labels, preds, zero_division=0)),
        'f1'       : float(f1_score(labels, preds, zero_division=0)),
        'cm'       : confusion_matrix(labels, preds),
    }


def print_scores(title, s):
    print(f"\n{title}")
    print(f"  Accuracy  : {s['accuracy']:.2f}%")
    print(f"  Precision : {s['precision']:.3f}")
    print(f"  Recall    : {s['recall']:.3f}")
    print(f"  F1        : {s['f1']:.3f}")
    print(f"  Confusion matrix (rows=actual, cols=predicted):\n{s['cm']}")


def evaluate():
    probs, labels = get_probabilities()

    majority = max(labels.mean(), 1 - labels.mean()) * 100              # always-guess-majority accuracy
    pr_auc   = float(average_precision_score(labels, probs))

    print("\n" + "=" * 50)
    print("TEST SET RESULTS (held-out machines)")
    print("=" * 50)
    print(f"Test sequences: {len(labels)} | At-risk share: {labels.mean() * 100:.1f}%")
    print(f"Majority-class accuracy: {majority:.2f}%")

    default = score(probs, labels, 0.5)
    tuned   = score(probs, labels, THRESHOLD)
    print_scores("At default threshold 0.5:", default)
    print_scores(f"At tuned threshold {THRESHOLD}:", tuned)
    print(f"\n✅ Test PR-AUC: {pr_auc:.3f}")

    return default, tuned, pr_auc, majority


# ─────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────

if __name__ == "__main__":
    mlflow.set_experiment("LSTM_Risk_Predictor")

    with mlflow.start_run(run_name="LSTM_Test_Evaluation"):
        default, tuned, pr_auc, majority = evaluate()

        mlflow.log_metrics({
            'test_accuracy'        : tuned['accuracy'],
            'test_precision'       : tuned['precision'],
            'test_recall'          : tuned['recall'],
            'test_f1'              : tuned['f1'],
            'test_pr_auc'          : pr_auc,
            'test_accuracy_at_0.5' : default['accuracy'],
            'test_recall_at_0.5'   : default['recall'],
            'majority_accuracy'    : majority,
            'threshold'            : THRESHOLD,
        })