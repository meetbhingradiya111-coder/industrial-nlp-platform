import torch
import json
from transformers import DistilBertTokenizer
from models.bert_classifier.model import BERTFailureClassifier

def load_model():
    """Load trained BERT model"""

    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(
        'distilbert-base-uncased'
    )

    # Load model
    model = BERTFailureClassifier(num_classes=6)
    model.load_state_dict(
        torch.load(
            'models/bert_classifier/best_model.pt',
            map_location=device
        )
    )
    model.to(device)
    model.eval()

    # Load label mapping
    with open('data/clean/failure_mapping.json') as f:
        mapping = json.load(f)

    return model, tokenizer, mapping, device


def predict_single(log_text):
    """
    Predict failure type for one log entry
    """

    model, tokenizer, mapping, device = load_model()

    # Tokenize input
    encoding = tokenizer(
        log_text,
        max_length     = 128,
        padding        = 'max_length',
        truncation     = True,
        return_tensors = 'pt'
    )

    # Move to device
    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs   = torch.softmax(outputs, dim=1)
        pred    = torch.argmax(probs, dim=1).item()
        conf    = probs[0][pred].item() * 100

    failure_type = mapping[str(pred)]

    return {
        'failure_type' : failure_type,
        'confidence'   : f"{conf:.1f}%",
        'label'        : pred
    }


# ── TEST WITH SAMPLE LOGS ────────────
if __name__ == "__main__":

    test_logs = [
        "Loud grinding noise from motor shaft bearing",
        "Motor overheating sparks from electrical panel",
        "Oil leak detected near base seal of shaft",
        "Routine inspection completed all normal",
        "Minor vibration noticed will monitor",
    ]

    print("\n── BERT PREDICTIONS ──────────────\n")

    for log in test_logs:
        result = predict_single(log)
        print(f"Log   : {log}")
        print(f"Pred  : {result['failure_type']}")
        print(f"Conf  : {result['confidence']}")
        print()