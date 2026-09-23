import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from models.autoencoder.model import LogAutoencoder
from models.autoencoder.dataset import LogAnomalyDataset, get_bert_embedding

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

CONFIG = {
    'val_path'   : 'data/clean/val.csv',
    'test_path'  : 'data/clean/test.csv',
    'models'     : {
        'ORIGINAL' : 'models/autoencoder/saved/best_model_original_statedict.pt',
        'AUGMENTED': 'models/autoencoder/saved/best_model.pt',
    },
    'input_dim'    : 768,
    'encoding_dim' : 64,
    'dropout'      : 0.2,
    'batch_size'   : 32,
    'percentiles'  : [50, 75, 90, 95, 97, 98, 99, 99.5],
}

NATURAL_LANGUAGE_TESTS = {
    "Test 1": "Routine maintenance inspection completed. Machine is operating normally. Temperature, vibration, pressure, and motor performance are within the expected range. No abnormal noise or mechanical issues were observed.",
    "Test 2": "The technician inspected the motor and gearbox after the morning shift. Everything appeared to be functioning as expected, with stable operating temperature and no unusual vibration, noise, leakage, or pressure fluctuations.",
    "Test 3": "During the scheduled inspection, the technician checked the motor, gearbox, vibration level, operating temperature, and pressure readings. All measurements remained within the expected operating range and no abnormal mechanical behavior was observed.",
    "Test 4": "Preventive maintenance was completed on the machine today. The motor, bearings, and gearbox were inspected and found to be operating normally. No signs of wear, overheating, abnormal vibration, or unusual noise were detected.",
    "Test 5": "The equipment was monitored throughout the shift and continued to operate smoothly. Temperature and pressure remained stable, vibration levels were normal, and no mechanical or electrical issues were reported.",
}


# ─────────────────────────────────────
# 2. DEVICE
# ─────────────────────────────────────

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device} ✅\n")


# ─────────────────────────────────────
# 3. HELPER: LOAD A MODEL
# ─────────────────────────────────────

def load_model(checkpoint_path):
    model = LogAutoencoder(
        input_dim=CONFIG['input_dim'],
        encoding_dim=CONFIG['encoding_dim'],
        dropout=CONFIG['dropout']
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


# ─────────────────────────────────────
# 4. HELPER: RECONSTRUCTION ERRORS
# ─────────────────────────────────────

def compute_errors(model, loader):
    criterion = nn.MSELoss(reduction='none')
    errors, labels = [], []
    with torch.no_grad():
        for features, lbl in loader:
            features = features.to(device)
            reconstructed = model(features)
            batch_errors = criterion(reconstructed, features).mean(dim=1)
            errors.extend(batch_errors.cpu().numpy())
            labels.extend(lbl.numpy())
    return np.array(errors), np.array(labels)


# ─────────────────────────────────────
# 5. ANALYZE ONE MODEL AT ALL PERCENTILES
# ─────────────────────────────────────

def analyze_model(model_name, checkpoint_path):
    print("\n" + "="*60)
    print(f"MODEL: {model_name}")
    print("="*60)

    model = load_model(checkpoint_path)

    # ── Normal validation error distribution ──
    normal_val_dataset = LogAnomalyDataset(CONFIG['val_path'], filter_normal_only=True)
    normal_val_loader = DataLoader(normal_val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    normal_errors, _ = compute_errors(model, normal_val_loader)

    print(f"\nNormal validation error distribution ({len(normal_errors)} samples):")
    print(f"  Mean   : {normal_errors.mean():.6f}")
    print(f"  Std    : {normal_errors.std():.6f}")
    print(f"  Min    : {normal_errors.min():.6f}")
    print(f"  Max    : {normal_errors.max():.6f}")

    percentile_values = {}
    for p in CONFIG['percentiles']:
        val = np.percentile(normal_errors, p)
        percentile_values[p] = val
        print(f"  {p}th percentile: {val:.6f}")

    # ── Full test set (normal + failure) ──
    test_dataset = LogAnomalyDataset(CONFIG['test_path'], filter_normal_only=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_errors, test_labels = compute_errors(model, test_loader)

    # ── Evaluate every percentile threshold against test set ──
    print(f"\n{'Pctl':>6} | {'Thresh':>10} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Acc':>6} | {'FPR':>6} | {'FNR':>6} | {'Normal FP Rate':>14}")
    print("-" * 100)

    results = []
    for p in CONFIG['percentiles']:
        threshold = percentile_values[p]
        preds = (test_errors > threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, preds, average='binary', zero_division=0
        )
        acc = accuracy_score(test_labels, preds)

        # False positive rate: normal logs incorrectly flagged as anomaly
        normal_mask = test_labels == 0
        fpr = (preds[normal_mask] == 1).sum() / normal_mask.sum() if normal_mask.sum() > 0 else 0

        # False negative rate: failure logs incorrectly missed
        failure_mask = test_labels == 1
        fnr = (preds[failure_mask] == 0).sum() / failure_mask.sum() if failure_mask.sum() > 0 else 0

        # Separately: false-positive rate on the NORMAL VALIDATION set itself
        normal_val_fp_rate = (normal_errors > threshold).sum() / len(normal_errors)

        print(f"{p:>6} | {threshold:>10.6f} | {precision:>6.4f} | {recall:>6.4f} | {f1:>6.4f} | {acc:>6.4f} | {fpr:>6.4f} | {fnr:>6.4f} | {normal_val_fp_rate:>14.4f}")

        results.append({
            'percentile': p, 'threshold': threshold, 'precision': precision,
            'recall': recall, 'f1': f1, 'accuracy': acc, 'fpr': fpr, 'fnr': fnr
        })

    # ── Natural-language test logs: where do they fall in the distribution? ──
    print(f"\nNatural-language test logs — reconstruction error and percentile rank:")
    criterion = nn.MSELoss()
    nl_errors = {}
    for label, text in NATURAL_LANGUAGE_TESTS.items():
        embedding = get_bert_embedding(text)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            reconstructed = model(embedding_tensor)
            error = criterion(reconstructed, embedding_tensor).item()

        # What percentile of the NORMAL distribution does this error fall at?
        percentile_rank = (normal_errors < error).sum() / len(normal_errors) * 100
        nl_errors[label] = error

        print(f"  {label}: error={error:.6f}  →  falls at {percentile_rank:.1f}th percentile of normal distribution")

    return {
        'normal_errors': normal_errors,
        'results': results,
        'nl_errors': nl_errors,
    }


# ─────────────────────────────────────
# 6. RUN FOR BOTH MODELS
# ─────────────────────────────────────

if __name__ == "__main__":
    all_results = {}
    for name, path in CONFIG['models'].items():
        all_results[name] = analyze_model(name, path)

    # ── Final comparison table ──
    print("\n\n" + "="*60)
    print("FINAL COMPARISON: NATURAL-LANGUAGE TEST ERRORS")
    print("="*60)
    print(f"{'Test':>8} | {'ORIGINAL':>12} | {'AUGMENTED':>12}")
    for test_name in NATURAL_LANGUAGE_TESTS.keys():
        orig = all_results['ORIGINAL']['nl_errors'][test_name]
        aug = all_results['AUGMENTED']['nl_errors'][test_name]
        print(f"{test_name:>8} | {orig:>12.6f} | {aug:>12.6f}")