import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertModel

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

CONFIG = {
    'bert_name'    : 'distilbert-base-uncased',
    'max_length'   : 128,
    'embed_batch'  : 64,
    'failure_types': [
        'Bearing Failure',
        'Electrical Fault',
        'Lubrication Failure',
        'Mechanical Fault',
    ],
}


# ─────────────────────────────────────
# 2. DEVICE
# ─────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")                                      # M2 GPU
    return torch.device("cpu")                                          # fallback


# ─────────────────────────────────────
# 3. EMBEDDING CACHE
# ─────────────────────────────────────

def embed_unique_texts(texts, device):
    """Embeds each unique log text once, returns {text: 768-dim vector}"""
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['bert_name'])    # load tokenizer
    bert = DistilBertModel.from_pretrained(CONFIG['bert_name']).to(device)  # load BERT
    bert.eval()

    unique = sorted(set(str(t) for t in texts))                         # unique texts only
    print(f"  ✅ Embedding {len(unique)} unique log texts...")
    cache = {}

    with torch.no_grad():
        for start in range(0, len(unique), CONFIG['embed_batch']):
            batch = unique[start:start + CONFIG['embed_batch']]         # current batch
            tokens = tokenizer(
                batch,
                max_length     = CONFIG['max_length'],
                padding        = 'max_length',
                truncation     = True,
                return_tensors = 'pt'
            )
            output = bert(
                tokens['input_ids'].to(device),
                tokens['attention_mask'].to(device)
            )
            emb = output.last_hidden_state[:, 0, :].cpu().numpy()       # CLS embedding
            for text, vec in zip(batch, emb):
                cache[text] = vec                                       # store per text

    return cache


# ─────────────────────────────────────
# 4. SEQUENCE DATASET
# ─────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Creates sliding windows of N logs per machine
    Input  : sequence of BERT embeddings (N, 768)
    Output : 1 if a failure occurs in the next `horizon` logs, else 0
    """

    def __init__(self, csv_path, sequence_len=10, horizon=1):
        self.sequence_len = sequence_len
        self.horizon      = horizon
        self.sequences    = []
        self.labels       = []

        df = pd.read_csv(csv_path)                                      # read csv
        df['date'] = pd.to_datetime(df['date'])                         # parse dates
        df = df.sort_values(['machine_id', 'date']).reset_index(drop=True)

        print(f"Loading {csv_path}...")
        print(f"Total records: {len(df)}")

        cache = embed_unique_texts(df['clean_log'].tolist(), get_device())  # embed once per text
        df['is_failure'] = df['failure_type'].isin(
            CONFIG['failure_types']
        ).astype(int)                                                   # 1 = failure log

        print("Creating sequences...")
        for machine_id in df['machine_id'].unique():
            machine_df = df[df['machine_id'] == machine_id].reset_index(drop=True)
            vectors    = np.stack([cache[str(t)] for t in machine_df['clean_log']])  # (logs, 768)
            failures   = machine_df['is_failure'].values                # failure flags

            for i in range(len(machine_df) - sequence_len - horizon + 1):
                seq_embeddings = vectors[i:i + sequence_len]            # window of embeddings
                future         = failures[i + sequence_len : i + sequence_len + horizon]
                label          = int(future.max())                      # failure within horizon

                self.sequences.append(seq_embeddings)
                self.labels.append(label)

        at_risk = sum(self.labels)
        print(f"Total sequences created: {len(self.sequences)}")
        print(f"At-risk sequences : {at_risk} ({at_risk / max(len(self.labels), 1) * 100:.1f}%)")
        print(f"Normal sequences  : {len(self.labels) - at_risk}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = torch.tensor(self.sequences[index], dtype=torch.float32)
        label    = torch.tensor(self.labels[index], dtype=torch.long)
        return {'sequence': sequence, 'label': label}