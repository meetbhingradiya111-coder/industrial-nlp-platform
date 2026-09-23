import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertModel

class SequenceDataset(Dataset):
    """
    Creates sequences of last N logs per machine
    Input  : sequence of BERT embeddings
    Output : risk label (0 = safe, 1 = at risk)
    """

    def __init__(self, csv_path, sequence_len=10):
        self.sequence_len = sequence_len
        self.sequences    = []
        self.labels       = []

        # Load data
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['machine_id', 'date'])

        print(f"Loading {csv_path}...")
        print(f"Total records: {len(df)}")

        # Load BERT for embeddings
        print("Loading BERT for embeddings...")
        tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased'
        )

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        bert = DistilBertModel.from_pretrained(
            'distilbert-base-uncased'
        ).to(device)
        bert.eval()

        # Get embedding for each log
        print("Generating embeddings...")
        embeddings = []

        with torch.no_grad():
            for text in df['clean_log'].tolist():
                tokens = tokenizer(
                    str(text),
                    max_length     = 128,
                    padding        = 'max_length',
                    truncation     = True,
                    return_tensors = 'pt'
                )
                input_ids      = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)

                output    = bert(input_ids, attention_mask)
                embedding = output.last_hidden_state[:, 0, :]
                embeddings.append(
                    embedding.cpu().numpy().squeeze()
                )

        df['embedding'] = embeddings

        # Create failure flag
        # 1 = failure, 0 = normal/warning
        failure_types = [
            'Bearing Failure',
            'Electrical Fault',
            'Lubrication Failure',
            'Mechanical Fault'
        ]
        df['is_failure'] = df['failure_type'].apply(
            lambda x: 1 if x in failure_types else 0
        )

        # Create sequences per machine
        print("Creating sequences...")
        for machine_id in df['machine_id'].unique():
            machine_df = df[df['machine_id'] == machine_id]
            machine_df = machine_df.reset_index(drop=True)

            for i in range(len(machine_df) - sequence_len):
                # Get sequence of embeddings
                seq = machine_df.iloc[i:i+sequence_len]
                seq_embeddings = np.stack(
                    seq['embedding'].values
                )

                # Label = is there a failure
                # in next log after sequence?
                next_log = machine_df.iloc[i + sequence_len]
                label    = int(next_log['is_failure'])

                self.sequences.append(seq_embeddings)
                self.labels.append(label)

        print(f"Total sequences created: {len(self.sequences)}")
        print(f"Failure sequences : {sum(self.labels)}")
        print(f"Normal sequences  : {len(self.labels) - sum(self.labels)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = torch.tensor(
            self.sequences[index],
            dtype=torch.float32
        )
        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )
        return {'sequence': sequence, 'label': label}