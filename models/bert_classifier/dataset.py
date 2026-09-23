# models/bert_classifier/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import DistilBertTokenizer

class MaintenanceLogDataset(Dataset):
    """
    Custom Dataset for Maintenance Logs
    
    Converts text logs into BERT tokens
    Returns tensors ready for training
    """

    def __init__(self, csv_path, tokenizer, max_length=128):
        """
        csv_path   : Path to train.csv or val.csv
        tokenizer  : BERT tokenizer
        max_length : Max number of tokens per log
        """

        # Load data
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Get logs and labels
        self.logs   = self.data['clean_log'].tolist()
        self.labels = self.data['failure_label'].tolist()

        print(f"Dataset loaded: {len(self.logs)} records")
        print(f"Columns: {list(self.data.columns)}")

    def __len__(self):
        """Return total number of records"""
        return len(self.logs)

    def __getitem__(self, index):
        """
        Return one record as tensors
        Called automatically during training
        """

        # Get one log and its label
        log   = str(self.logs[index])
        label = int(self.labels[index])

        # Tokenize the log using BERT tokenizer
        encoding = self.tokenizer(
            log,
            max_length      = self.max_length,
            padding         = 'max_length',
            truncation      = True,
            return_tensors  = 'pt'
        )

        return {
            'input_ids'      : encoding['input_ids'].squeeze(),
            'attention_mask' : encoding['attention_mask'].squeeze(),
            'label'          : torch.tensor(label, dtype=torch.long)
        }