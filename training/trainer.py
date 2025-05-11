import torch
from config import GPT_CONFIG_124M, TRAINING_CONFIG, EVAL_CONFIG, OPTIMIZER_CONFIG
from models.gpt_model import GPTModel
from training.train import train_model_simple, train_loader, val_loader
from utils.utils import get_tokenizer, plot_losses
from utils.model_utils import save_model
from constants import device

class Trainer:
    def __init__(self):
        # Set random seed
        torch.manual_seed(123)
        
        # Initialize model
        print("Initializing model...")
        self.model = GPTModel(GPT_CONFIG_124M)
        self.model.to(device)
        
        # Initialize optimizer
        print("Setting up optimizer...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=OPTIMIZER_CONFIG["learning_rate"],
            weight_decay=OPTIMIZER_CONFIG["weight_decay"]
        )
        
        # Get tokenizer
        self.tokenizer = get_tokenizer()
        
        # Training components
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def train(self):
        """Execute the complete training process"""
        print("Starting training...")
        train_losses, val_losses, tokens_seen = train_model_simple(
            self.model, self.train_loader, self.val_loader, self.optimizer, device,
            num_epochs=TRAINING_CONFIG["num_epochs"],
            eval_freq=EVAL_CONFIG["eval_freq"],
            eval_iter=EVAL_CONFIG["eval_iter"],
            start_context=TRAINING_CONFIG["start_context"],
            tokenizer=self.tokenizer
        )
        
        # Plot training results
        print("Plotting training results...")
        epochs_tensor = torch.linspace(0, TRAINING_CONFIG["num_epochs"], len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        
        # Save model
        print("Saving model...")
        save_path = save_model(self.model, self.optimizer)
        print(f"Training complete! Model saved at: {save_path}")
        
        return train_losses, val_losses, tokens_seen, save_path
