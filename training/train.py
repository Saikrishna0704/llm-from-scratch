from evaluation.evaluate import calc_loss_batch, evaluate_model
from utils.utils import generate_and_print_sample
from config import GPT_CONFIG_124M, TRAINING_CONFIG
from data.data_loader import Data_loader

dl = Data_loader()
train_data = dl.train_data
val_data = dl.val_data

train_loader  = dl.create_dataloader_v1(
    train_data,
    batch_size=TRAINING_CONFIG["batch_size"],
    max_length= GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    shuffle = TRAINING_CONFIG["shuffle"]["train"],
    drop_last=TRAINING_CONFIG["drop_last"]["train"],
    num_workers = TRAINING_CONFIG["num_workers"]
)

val_loader  = dl.create_dataloader_v1(
    val_data,
    batch_size=TRAINING_CONFIG["batch_size"],
    max_length= GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    shuffle = TRAINING_CONFIG["shuffle"]["val"],
    drop_last=TRAINING_CONFIG["drop_last"]["val"],
    num_workers = TRAINING_CONFIG["num_workers"]
)

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step =0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch,model,device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)
            print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}")
        
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen 


