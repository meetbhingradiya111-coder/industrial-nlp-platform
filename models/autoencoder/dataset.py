import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel

# ─────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────

CONFIG = {
    'max_length'   : 128,   # Max tokens per log
    'embedding_dim': 768,   # DistilBERT's hidden size
}


# ─────────────────────────────────────
# 2. DEVICE DETECTION (M2 Mac = mps)
# ─────────────────────────────────────

if torch.backends.mps.is_available():
    _device = torch.device("mps")
elif torch.cuda.is_available():
    _device = torch.device("cuda")
else:
    _device = torch.device("cpu")


# ─────────────────────────────────────
# 3. SHARED BERT EMBEDDING MODEL
# ─────────────────────────────────────
# NOTE: This embedding function is used ONLY by the autoencoder.
# BERT classifier and LSTM have their own separate embedding logic
# and are NOT affected by this change.

_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
_embed_bert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(_device)
_embed_bert.eval()


def get_bert_embedding(text: str):
    """
    Converts text into a 768-dim BERT embedding using MEAN POOLING
    (average of all real token embeddings, excluding padding),
    rather than just the [CLS] token.

    Mean pooling is less sensitive to sentence length/structure than
    [CLS] alone, which should generalize better across short templated
    logs and longer natural-language logs describing the same meaning.
    """
    tokens = _tokenizer(
        str(text),
        max_length=CONFIG['max_length'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = tokens['input_ids'].to(_device)
    attention_mask = tokens['attention_mask'].to(_device)

    with torch.no_grad():
        output = _embed_bert(input_ids, attention_mask)
        token_embeddings = output.last_hidden_state   # (1, seq_len, 768)

    # Expand attention mask to match embedding dimensions
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # Sum only the REAL (non-padding) token embeddings
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)

    # Count real tokens (avoid dividing by zero)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

    # Mean = sum of real token embeddings / count of real tokens
    mean_embedding = sum_embeddings / sum_mask

    return mean_embedding.cpu().numpy().squeeze()


# ─────────────────────────────────────
# 4. DATASET CLASS
# ─────────────────────────────────────

class LogAnomalyDataset(Dataset):
    """
    Converts maintenance logs into mean-pooled BERT embeddings for the autoencoder.
    """

    def __init__(self, csv_path, filter_normal_only=False):
        self.data = pd.read_csv(csv_path)

        if filter_normal_only:
            self.data = self.data[self.data['is_failure'] == 0].reset_index(drop=True)

        self.texts = self.data['log_entry'].astype(str).tolist()

        print(f"Generating mean-pooled BERT embeddings for {len(self.texts)} logs...")
        self.features = [get_bert_embedding(text) for text in self.texts]

        self.labels = self.data['is_failure'].astype(int).tolist()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_vector = torch.tensor(self.features[idx], dtype=torch.float32)
        label = self.labels[idx]
        return feature_vector, label