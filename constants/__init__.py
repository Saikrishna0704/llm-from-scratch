import torch

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_file_path = "/Users/saikrishna/Desktop/ML/LLMfromScratch/the-verdict.txt"
