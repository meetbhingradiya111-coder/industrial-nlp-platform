import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
import json
import mlflow
from models.autoencoder.model import LogAutoencoder
from models.autoencoder.dataset import LogAnomalyDataset, get_bert_embedding

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

CONFIG = {
    'val_path'    : 'data/clean/val.csv',
    'test_path'   : 'data/clean/test.csv',
    'model_path'  : 'models/autoencoder/saved/best_model.pt',
    'input_dim'   : 768,
    'encoding_dim': 64,
    'dropout'     : 0.2,
    'batch_size'  : 32,
    'percentile'  : 97,   # Threshold = 97th percentile of NORMAL validation errors
    'threshold_save_path': 'models/autoencoder/saved/threshold.json',  # Threshold = 95th percentile of NORMAL validation errors
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
# 3. LOAD TRAINED MODEL
# ─────────────────────────────────────

print("\nLoading trained autoencoder...")

model = LogAutoencoder(
    input_dim=CONFIG['input_dim'],
    encoding_dim=CONFIG['encoding_dim'],
    dropout=CONFIG['dropout']
).to(device)

model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
model.eval()

print("Model loaded ✅")


# ─────────────────────────────────────
# 4. RECONSTRUCTION ERROR FUNCTION
# ─────────────────────────────────────

def compute_reconstruction_errors(model, loader, device):
    """Returns per-sample reconstruction error and true labels"""

    criterion = nn.MSELoss(reduction='none')
    errors = []
    true_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            reconstructed = model(features)
            batch_errors = criterion(reconstructed, features).mean(dim=1)
            errors.extend(batch_errors.cpu().numpy())
            true_labels.extend(labels.numpy())

    return np.array(errors), np.array(true_labels)


# ─────────────────────────────────────
# 5. CALCULATE STATISTICALLY JUSTIFIED THRESHOLD
# ─────────────────────────────────────

def calculate_threshold():
    """
    Computes reconstruction error threshold using the Nth percentile
    of errors on NORMAL-ONLY validation logs. This means: if we accept
    a 5% false-positive rate on genuinely normal logs, anything above
    this value is statistically unusual enough to flag.
    """

    print("\nEmbedding normal validation logs to calculate threshold...")
    normal_val_dataset = LogAnomalyDataset(CONFIG['val_path'], filter_normal_only=True)
    normal_val_loader = DataLoader(normal_val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    errors, _ = compute_reconstruction_errors(model, normal_val_loader, device)

    threshold = np.percentile(errors, CONFIG['percentile'])

    print(f"\nNormal validation error stats:")
    print(f"  Mean   : {errors.mean():.6f}")
    print(f"  Std    : {errors.std():.6f}")
    print(f"  Min    : {errors.min():.6f}")
    print(f"  Max    : {errors.max():.6f}")
    print(f"  {CONFIG['percentile']}th percentile (threshold): {threshold:.6f}")

    # Save the ACTUAL calculated threshold — never hand-typed elsewhere
    threshold_data = {
        "anomaly_threshold": float(threshold),
        "percentile": CONFIG['percentile']
    }
    with open(CONFIG['threshold_save_path'], 'w') as f:
        json.dump(threshold_data, f, indent=2)
    print(f"\n✅ Threshold saved to {CONFIG['threshold_save_path']}")

    return threshold


# ─────────────────────────────────────
# 6. EVALUATE ON FULL TEST SET
# ─────────────────────────────────────

def evaluate_test_set(threshold):
    """Runs the model on the full test set (normal + failure) using the calculated threshold"""

    print("\nEmbedding full test set (normal + failure logs)...")
    test_dataset = LogAnomalyDataset(CONFIG['test_path'], filter_normal_only=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    errors, true_labels = compute_reconstruction_errors(model, test_loader, device)

    normal_errors = errors[true_labels == 0]
    failure_errors = errors[true_labels == 1]

    print(f"\nAvg reconstruction error (normal logs)  : {normal_errors.mean():.6f}")
    print(f"Avg reconstruction error (failure logs) : {failure_errors.mean():.6f}")

    auc_score = roc_auc_score(true_labels, errors)
    print(f"\nROC-AUC Score: {auc_score:.4f}")

    predictions = (errors > threshold).astype(int)

    print(f"\nUsing threshold: {threshold:.6f}")
    print("\nClassification Report (0=Normal, 1=Failure):")
    print(classification_report(true_labels, predictions, digits=4))

    report_dict = classification_report(true_labels, predictions, digits=4, output_dict=True)
    return auc_score, report_dict


# ─────────────────────────────────────
# 7. TEST WITH NATURAL-LANGUAGE LOGS
# ─────────────────────────────────────

def test_natural_language_logs(threshold):
    """
    Sanity check with realistic, naturally-phrased logs (not the short
    templated style used in training) to verify the model generalizes.
    """

    test_logs = {
        "Normal"       : "Routine inspection completed. Machine is operating normally with stable temperature, pressure, vibration, and motor performance. No abnormal noise or mechanical issues detected.",
        "Pre-failure"  : "Increasing vibration detected in the gearbox during operation. Temperature is slightly above normal and unusual mechanical noise was observed.",
        "Bearing"      : "Abnormal bearing noise detected during operation. Excessive vibration observed near the rotating shaft and bearing temperature is increasing rapidly.",
        "Severe gearbox": "Gearbox producing unusual grinding noise with severe vibration during operation. Gear temperature is increasing and gear movement appears irregular.",
        "Extreme"      : "Multiple severe machine conditions detected simultaneously. Extreme vibration and overheating observed with sudden hydraulic pressure loss, loud grinding noise, repeated unexpected shutdowns, and abnormal motor performance.",
    }

    print("\n" + "="*50)
    print("NATURAL-LANGUAGE LOG SANITY CHECK")
    print("="*50)

    criterion = nn.MSELoss()

    for label, text in test_logs.items():
        embedding = get_bert_embedding(text)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            reconstructed = model(embedding_tensor)
            error = criterion(reconstructed, embedding_tensor).item()

        is_anomaly = error > threshold
        status = "🔴 ANOMALY" if is_anomaly else "🟢 NORMAL"

        print(f"\n[{label}] {status}")
        print(f"  Error: {error:.6f}  |  Threshold: {threshold:.6f}")


# ─────────────────────────────────────
# 8. RUN FULL EVALUATION
# ─────────────────────────────────────

if __name__ == "__main__":
    mlflow.set_experiment("Autoencoder_Anomaly_Detector_BERT")

    with mlflow.start_run(run_name="Autoencoder_Evaluation"):
        threshold = calculate_threshold()
        auc, report = evaluate_test_set(threshold)
        test_natural_language_logs(threshold)

        mlflow.log_metric("threshold", threshold)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("failure_precision", report["1"]["precision"])
        mlflow.log_metric("failure_recall", report["1"]["recall"])
        mlflow.log_metric("failure_f1", report["1"]["f1-score"])
        mlflow.log_metric("normal_precision", report["0"]["precision"])
        mlflow.log_metric("normal_recall", report["0"]["recall"])

        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        print(f"Calculated Threshold : {threshold:.6f}")
        print(f"Test Set ROC-AUC     : {auc:.4f}")
        print(f"\n⚠️  Update CONFIG['anomaly_threshold'] in api/main.py to: {threshold:.6f}")