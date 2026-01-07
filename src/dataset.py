import os 
import torch 
from torch.utils.data import Dataset

class IMDBDataset(Dataset):
    def __init__(self,data,tokenizer, max_length =128);
    self.data = data 
    self.tokenizer = tokenizer
    self.max_length = max_length

    def __len__(self):
        retuen len(self.data)

    def __getitem__(self,idx):
        text = self.data.iloc[idx]["review"]
        leabel = int(self.data.iloc[idx]["label"])

        encoded = self.tokenizer(
            text,
            truncation =Ture,
            padding = "max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeexe(0)
        return input_ids,label
