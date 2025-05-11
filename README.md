# GPT Model from Scratch

#This is not the final implemention. New changes and additions are done as necessary.

A PyTorch implementation of a GPT-style language model trained from scratch.

## Project Structure

```
LLMfromScratch/
├── __main__.py        # Main entry point
├── config/            # Configuration directory
├── data/              # Data handling
├── evaluation/        # Evaluation metrics
├── models/            # Model architecture
├── training/          # Training related code
├── utils/             # Utility functions
├── constants/         # Constants and paths
├── outputs/           # Saved models and outputs
├── requirements.txt   # Project dependencies
└── the-verdict.txt    # Training data
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LLMfromScratch.git
cd LLMfromScratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the model:
```bash
python __main__.py
```

The model will be saved in the `outputs` directory with a timestamp.

## Configuration

Model and training parameters can be modified in `config.py`:
- Model architecture (GPT_CONFIG_124M)
- Training parameters (TRAINING_CONFIG)
- Evaluation settings (EVAL_CONFIG)
- Optimizer settings (OPTIMIZER_CONFIG)

## Requirements

- Python 3.8+
- PyTorch
- tiktoken
- matplotlib

See `requirements.txt` for full list of dependencies.
