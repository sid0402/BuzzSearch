import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import json

class FullDataset(Dataset):
    def __init__(self, html_path, reddit_path):
        self.html_path = html_path
        self.reddit_path = reddit_path

        self.html_data = self.load_jsonl_data(html_path)[0]
        self.reddit_data = self.load_jsonl_data(reddit_path)[0]

        self.max_query_length = 120

        data = []
        for row in self.html_data:
            query = f"query: {row['anchor']} {row['context']}"
            query = query.split(" ")[:self.max_query_length]
            query = " ".join(query)
            new_row = {
                "query": query,
                "passage": f"passage: {row['html_cleaned']['passages'][0]}",
                "id": row['url'],
                "type": "html"
            }
            data.append(new_row)
        
        for row in self.reddit_data:
            query = f"query: {row['anchor']} {row['context']}"
            query = query.split(" ")[:self.max_query_length]
            query = " ".join(query)
            new_row = {
                "query": query,
                "passage": f"passage: {row['document_text']['passages'][0]}",
                "id": row['id'],
                "type": "reddit"
            }
            data.append(new_row)
        
        self.data = data
        

    def load_jsonl_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
