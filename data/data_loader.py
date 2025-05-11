import torch
from torch.utils.data import Dataset, DataLoader
from utils.utils import get_tokenizer
from config import TRAINING_CONFIG
from constants import data_file_path

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

class Data_loader:
    def __init__(self):
        with open(data_file_path, "r", encoding = "utf-8") as file:
            text_data = file.read()
        self.tokenizer = get_tokenizer()
        self.split_idx = int(TRAINING_CONFIG["train_ratio"] * len(text_data))
        self.train_data = text_data[:self.split_idx]
        self.val_data = text_data[self.split_idx:]

    def create_dataloader_v1(self,txt, batch_size, max_length,
                            stride, shuffle, drop_last, 
                            num_workers):

        dataset = GPTDatasetV1(txt, self.tokenizer, max_length, stride)
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,
                                drop_last=drop_last,num_workers=num_workers)
        return dataloader
    
    