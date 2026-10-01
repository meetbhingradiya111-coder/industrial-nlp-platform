import json
import os
import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel

from models.lstm_predictor.model import LSTMRiskPredictor

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

CONFIG = {
    'data_path'         : 'data/clean/maintenance_logs_clean.csv',
    'lstm_model_path'   : 'models/lstm_predictor/best_model.pt',
    'threshold_path'    : 'models/lstm_predictor/threshold.json',
    'default_threshold' : 0.5,
    'sequence_len'      : 10,
    'hidden_size'       : 128,
    'num_layers'        : 2,
    'dropout'           : 0.3,
    'max_length'        : 128,
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

print(f"LSTM predictor using device: {device} ✅")


# ─────────────────────────────────────
# 3. LOAD RISK THRESHOLD
# ─────────────────────────────────────

if os.path.exists(CONFIG['threshold_path']):
    with open(CONFIG['threshold_path'], 'r') as f:
        THRESHOLD = json.load(f)['threshold']                           # read saved threshold
    print(f"✅ Risk threshold loaded: {THRESHOLD}")
else:
    THRESHOLD = CONFIG['default_threshold']                             # fall back to default
    print(f"⚠️  threshold.json not found, using default: {THRESHOLD}")


# ─────────────────────────────────────
# 4. LOAD BERT (for generating embeddings only, not classification)
# ─────────────────────────────────────

print("\nLoading DistilBERT for embeddings...")

embed_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
embed_bert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
embed_bert.eval()

print("✅ Embedding model loaded")


# ─────────────────────────────────────
# 5. LOAD TRAINED LSTM MODEL
# ─────────────────────────────────────

print("\nLoading LSTM risk predictor...")

lstm_model = LSTMRiskPredictor(
    input_size=768,
    hidden_size=CONFIG['hidden_size'],
    num_layers=CONFIG['num_layers'],
    num_classes=2,
    dropout=CONFIG['dropout']
).to(device)

lstm_model.load_state_dict(
    torch.load(CONFIG['lstm_model_path'], map_location=device)
)
lstm_model.eval()

print("✅ LSTM loaded")


# ─────────────────────────────────────
# 6. LOAD FULL LOG HISTORY
# ─────────────────────────────────────

full_data = pd.read_csv(CONFIG['data_path'])                            # read all logs
full_data['date'] = pd.to_datetime(full_data['date'])                   # parse dates


# ─────────────────────────────────────
# 7. EMBEDDING FUNCTION
# ─────────────────────────────────────

def get_embedding(text):
    """Converts a single log into a 768-dim BERT embedding"""

    tokens = embed_tokenizer(
        str(text),
        max_length=CONFIG['max_length'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = tokens['input_ids'].to(device)                          # token ids to device
    attention_mask = tokens['attention_mask'].to(device)                # mask to device

    with torch.no_grad():
        output = embed_bert(input_ids, attention_mask)                  # run BERT
        embedding = output.last_hidden_state[:, 0, :]                   # take CLS embedding

    return embedding.cpu().numpy().squeeze()


# ─────────────────────────────────────
# 8. RISK FROM A LIST OF LOGS
# ─────────────────────────────────────

def predict_from_logs(logs):
    """
    Predicts failure risk from a list of log strings
    (uses the most recent sequence_len logs)
    """

    if len(logs) < CONFIG['sequence_len']:
        return {
            'error': f"Need {CONFIG['sequence_len']} logs, got {len(logs)}."
        }

    recent = logs[-CONFIG['sequence_len']:]                             # keep last N logs
    embeddings = [get_embedding(text) for text in recent]               # embed each log
    sequence = np.stack(embeddings)                                     # shape (N, 768)

    # Add batch dimension: (1, sequence_len, 768)
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = lstm_model(sequence_tensor)                            # forward pass
        probs = torch.softmax(logits, dim=1)                            # class probabilities
        risk_prob = probs[0][1].item()                                  # probability of at risk

    return {
        'risk_probability': round(risk_prob, 4),
        'is_at_risk': bool(risk_prob >= THRESHOLD),
        'threshold': THRESHOLD,
        'logs_used': recent,
    }


# ─────────────────────────────────────
# 9. RISK FOR A MACHINE IN THE DATASET
# ─────────────────────────────────────

def predict_risk(machine_id: str):
    """
    Pulls the last N logs for a machine, embeds them,
    and predicts failure risk using the trained LSTM.
    """

    machine_logs = full_data[full_data['machine_id'] == machine_id]     # filter machine
    machine_logs = machine_logs.sort_values('date').reset_index(drop=True)

    if len(machine_logs) < CONFIG['sequence_len']:
        return {
            'error': f"Not enough history for {machine_id}. "
                     f"Found {len(machine_logs)} logs, need {CONFIG['sequence_len']}."
        }

    result = predict_from_logs(machine_logs['clean_log'].tolist())      # reuse log-based predictor
    result['machine_id'] = machine_id
    return result


# ─────────────────────────────────────
# 10. TEST WITH A SAMPLE MACHINE
# ─────────────────────────────────────

if __name__ == "__main__":
    sample_machine = full_data['machine_id'].iloc[0]
    result = predict_risk(sample_machine)
    print("\n── LSTM RISK PREDICTION ──────────────\n")
    print(result)