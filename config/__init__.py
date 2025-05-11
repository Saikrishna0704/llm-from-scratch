import torch

GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 256, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}


TRAINING_CONFIG = {
    "train_ratio" : 0.9,
    "batch_size" :2,
    "num_epochs": 5,
    "shuffle": {"train": True, "val": False},
    "drop_last": {"train": True, "val": True},
    "num_workers": 0,
    "start_context": "Every effort moves you"
}

OPTIMIZER_CONFIG = {
    "learning_rate": 0.0004,
    "weight_decay": 0.1,
}

EVAL_CONFIG = {
    "eval_freq": 5,
    "eval_iter": 5,
}

GENERATION_CONFIG = {
    "max_new_tokens": 100,
    "temperature": 0.0,
    "top_k": None,
}

