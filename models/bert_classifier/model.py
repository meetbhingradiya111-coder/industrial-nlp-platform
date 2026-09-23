# models/bert_classifier/model.py

import torch
import torch.nn as nn
from transformers import DistilBertModel

class BERTFailureClassifier(nn.Module):
    """
    BERT based classifier for maintenance logs

    Architecture:
        DistilBERT → Dropout → Dense → Output

    Input  : Tokenized log text
    Output : Failure category (0-5)
    """

    def __init__(self, num_classes=6, dropout=0.3):
        super(BERTFailureClassifier, self).__init__()

        # Load pretrained DistilBERT
        self.bert = DistilBertModel.from_pretrained(
            'distilbert-base-uncased'
        )

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # Classification head
        # 768 → num_classes
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model

        input_ids      : Token IDs from tokenizer
        attention_mask : Which tokens to attend to
        """

        # Pass through BERT
        bert_output = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask
        )

        # Get [CLS] token output
        # This represents the whole sentence
        cls_output = bert_output.last_hidden_state[:, 0, :]

        # Apply dropout
        dropped = self.dropout(cls_output)

        # Get class predictions
        logits = self.classifier(dropped)

        return logits