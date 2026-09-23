import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel

from models.lstm_predictor.model import LSTMRiskPredictor

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

CONFIG = {
    'data_path'      : 'data/clean/maintenance_logs_clean.csv',
    'lstm_model_path': 'models/lstm_predictor/best_model.pt',
    'sequence_len'   : 10,
    'hidden_size'    : 128,
    'num_layers'     : 2,
    'dropout'        : 0.3,
    'max_length'     : 128,
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
# 3. LOAD BERT (for generating embeddings only, not classification)
# ─────────────────────────────────────

print("\nLoading DistilBERT for embeddings...")

embed_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
embed_bert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
embed_bert.eval()

print("Embedding model loaded ✅")


# ─────────────────────────────────────
# 4. LOAD TRAINED LSTM MODEL
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

print("LSTM loaded ✅")


# ─────────────────────────────────────
# 5. LOAD FULL LOG HISTORY
# ─────────────────────────────────────

full_data = pd.read_csv(CONFIG['data_path'])
full_data['date'] = pd.to_datetime(full_data['date'])


# ─────────────────────────────────────
# 6. EMBEDDING FUNCTION
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
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    with torch.no_grad():
        output = embed_bert(input_ids, attention_mask)
        embedding = output.last_hidden_state[:, 0, :]

    return embedding.cpu().numpy().squeeze()


# ─────────────────────────────────────
# 7. RISK PREDICTION FUNCTION
# ─────────────────────────────────────

def predict_risk(machine_id: str):
    """
    Pulls the last N logs for a machine, embeds them,
    and predicts failure risk using the trained LSTM.
    """

    # Filter and sort this machine's logs by date
    machine_logs = full_data[full_data['machine_id'] == machine_id]
    machine_logs = machine_logs.sort_values('date').reset_index(drop=True)

    # Check we have enough history for a full sequence
    if len(machine_logs) < CONFIG['sequence_len']:
        return {
            'error': f"Not enough history for {machine_id}. "
                     f"Found {len(machine_logs)} logs, need {CONFIG['sequence_len']}."
        }

    # Take the most recent N logs
    recent_logs = machine_logs.tail(CONFIG['sequence_len'])

    # Generate embeddings for each log in the sequence
    embeddings = [get_embedding(text) for text in recent_logs['clean_log']]
    sequence = np.stack(embeddings)

    # Convert to tensor with batch dimension: (1, sequence_len, 768)
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

    # Run through LSTM
    with torch.no_grad():
        logits = lstm_model(sequence_tensor)
        probs = torch.softmax(logits, dim=1)
        risk_prob = probs[0][1].item()   # Probability of class 1 = at risk
        is_at_risk = risk_prob > 0.5

    return {
        'machine_id': machine_id,
        'risk_probability': round(risk_prob, 4),
        'is_at_risk': is_at_risk,
        'logs_used': recent_logs['clean_log'].tolist()
    }


# ─────────────────────────────────────
# 8. TEST WITH A SAMPLE MACHINE
# ─────────────────────────────────────

if __name__ == "__main__":
    sample_machine = full_data['machine_id'].iloc[0]
    result = predict_risk(sample_machine)
    print("\n── LSTM RISK PREDICTION ──────────────\n")
    print(result)