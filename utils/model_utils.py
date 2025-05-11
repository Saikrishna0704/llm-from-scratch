import torch
import os
from datetime import datetime

def save_model(model, optimizer, save_dir="outputs"):
    """
    Save model and optimizer state with timestamp.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        save_dir: Directory to save the model
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"model_{timestamp}.pth")
    
    # Save model and optimizer state
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "timestamp": timestamp
    }, save_path)
    
    return save_path

def load_model(model, optimizer, load_path):
    """
    Load model and optimizer state.
    
    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into
        load_path: Path to the saved model
    """
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint.get("timestamp") 