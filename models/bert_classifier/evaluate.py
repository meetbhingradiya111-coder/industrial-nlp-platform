# models/bert_classifier/evaluate.py

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
import json
import mlflow
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from models.bert_classifier.model import BERTFailureClassifier
from models.bert_classifier.dataset import MaintenanceLogDataset

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

CONFIG = {
    'test_path'     : 'data/clean/test.csv',
    'mapping_path'  : 'data/clean/failure_mapping.json',
    'model_path'    : 'models/bert_classifier/best_model.pt',
    'num_classes'   : 6,
    'max_length'    : 128,
    'dropout'       : 0.3,
    'batch_size'    : 16,
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
# 3. LOAD TOKENIZER AND TEST DATA
# ─────────────────────────────────────

print("\nLoading tokenizer and test set...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

test_dataset = MaintenanceLogDataset(
    CONFIG['test_path'],
    tokenizer,
    CONFIG['max_length']
)
test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False
)

print(f"Test samples: {len(test_dataset)}")

# ─────────────────────────────────────
# 4. LOAD LABEL MAPPING
# ─────────────────────────────────────

with open(CONFIG['mapping_path'], 'r') as f:
    failure_mapping = json.load(f)

class_names = [failure_mapping[str(i)] for i in range(CONFIG['num_classes'])]

# ─────────────────────────────────────
# 5. LOAD TRAINED MODEL
# ─────────────────────────────────────

print("\nLoading trained BERT classifier...")
model = BERTFailureClassifier(
    num_classes=CONFIG['num_classes'],
    dropout=CONFIG['dropout']
).to(device)
model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
model.eval()
print("Model loaded ✅")

# ─────────────────────────────────────
# 6. RUN INFERENCE ON TEST SET (evaluated ONCE)
# ─────────────────────────────────────

def evaluate():
    all_preds = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            confidences = probs.max(dim=1).values

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())

    return np.array(all_preds), np.array(all_labels), np.array(all_confidences)


if __name__ == "__main__":
    mlflow.set_experiment("BERT_Failure_Classifier")

    with mlflow.start_run(run_name="BERT_Evaluation"):
        print("\n" + "="*50)
        print("FINAL TEST SET EVALUATION (touched only once)")
        print("="*50)

        preds, labels, confidences = evaluate()

        accuracy = accuracy_score(labels, preds)
        print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%")
        print(f"Average Confidence: {confidences.mean()*100:.2f}%")

        print("\nPer-Class Metrics:")
        print(classification_report(labels, preds, target_names=class_names, digits=4))

        report = classification_report(labels, preds, target_names=class_names, digits=4, output_dict=True)

        print("\nConfusion Matrix:")
        cm = confusion_matrix(labels, preds)
        print(f"{'':22s}", end="")
        for name in class_names:
            print(f"{name[:10]:>12s}", end="")
        print()
        for i, row in enumerate(cm):
            print(f"{class_names[i]:22s}", end="")
            for val in row:
                print(f"{val:>12d}", end="")
            print()

        np.save('reports/confusion_matrix.npy', cm)
        print("\n✅ Confusion matrix saved to reports/confusion_matrix.npy")

        # Log overall metrics
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("avg_confidence", confidences.mean())
        mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])
        mlflow.log_metric("weighted_f1", report["weighted avg"]["f1-score"])

        # Log per-class precision/recall/f1
        for class_name in class_names:
            safe_name = class_name.replace(" ", "_")
            mlflow.log_metric(f"{safe_name}_precision", report[class_name]["precision"])
            mlflow.log_metric(f"{safe_name}_recall", report[class_name]["recall"])
            mlflow.log_metric(f"{safe_name}_f1", report[class_name]["f1-score"])

        mlflow.log_artifact("reports/confusion_matrix.npy")