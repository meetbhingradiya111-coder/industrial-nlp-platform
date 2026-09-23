import torch
import mlflow
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from models.lstm_predictor.model import LSTMRiskPredictor
from models.lstm_predictor.dataset import SequenceDataset

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

CONFIG = {
    'val_path'        : 'data/clean/val.csv',
    'lstm_model_path' : 'models/lstm_predictor/best_model.pt',
    'sequence_len'     : 10,
    'hidden_size'      : 128,
    'num_layers'       : 2,
    'dropout'          : 0.3,
    'batch_size'       : 32,
}


# ─────────────────────────────────────
# 2. DETECT DEVICE (M2 Mac = mps)
# ─────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device} ✅")


# ─────────────────────────────────────
# 3. LOAD VALIDATION SEQUENCES
# ─────────────────────────────────────

print("\nBuilding validation sequences (this embeds every log, may take a minute)...")

val_dataset = SequenceDataset(CONFIG['val_path'], sequence_len=CONFIG['sequence_len'])
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)


# ─────────────────────────────────────
# 4. LOAD TRAINED MODEL
# ─────────────────────────────────────

print("\nLoading trained LSTM...")

model = LSTMRiskPredictor(
    input_size=768,
    hidden_size=CONFIG['hidden_size'],
    num_layers=CONFIG['num_layers'],
    num_classes=2,
    dropout=CONFIG['dropout']
).to(device)

model.load_state_dict(torch.load(CONFIG['lstm_model_path'], map_location=device))
model.eval()

print("Model loaded ✅")


# ─────────────────────────────────────
# 5. RUN EVALUATION
# ─────────────────────────────────────

def evaluate():
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)

            outputs = model(sequences)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print("\n" + "="*50)
    print("CONFUSION MATRIX (rows=actual, cols=predicted)")
    print("="*50)
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)

    # Check what fraction of predictions are "normal" (class 0)
    pred_normal_ratio = all_preds.count(0) / len(all_preds)
    print(f"\n% of predictions that are 'Normal': {pred_normal_ratio*100:.1f}%")

    if pred_normal_ratio > 0.90:
        print("⚠️  Model is likely just predicting the majority class — not learning real patterns.")

    return report, pred_normal_ratio


if __name__ == "__main__":
    mlflow.set_experiment("LSTM_Risk_Predictor")

    with mlflow.start_run(run_name="LSTM_Evaluation"):
        report, pred_normal_ratio = evaluate()

        mlflow.log_metric("val_accuracy", report["accuracy"])
        mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])
        mlflow.log_metric("weighted_f1", report["weighted avg"]["f1-score"])
        mlflow.log_metric("pred_normal_ratio", pred_normal_ratio)

        if "0" in report:
            mlflow.log_metric("normal_precision", report["0"]["precision"])
            mlflow.log_metric("normal_recall", report["0"]["recall"])
        if "1" in report:
            mlflow.log_metric("at_risk_precision", report["1"]["precision"])
            mlflow.log_metric("at_risk_recall", report["1"]["recall"])