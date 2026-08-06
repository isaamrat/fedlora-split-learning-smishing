"""
shared/dataset.py — SMS dataset and dataloader helpers (shared by train_fedlora and train_split)
"""

import torch
from torch.utils.data import Dataset, DataLoader

MAX_LEN    = 128
LOCAL_BATCH = 16


class SMSDataset(Dataset):
    """Tokenises a list of SMS texts and pairs them with integer labels."""

    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt"
        )
        self.encodings["attention_mask"] = self.encodings["attention_mask"].bool()
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def make_loader(texts, labels, tokenizer, shuffle=True, batch_size=LOCAL_BATCH):
    """Return a DataLoader over an SMSDataset."""
    ds = SMSDataset(texts, labels, tokenizer)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
