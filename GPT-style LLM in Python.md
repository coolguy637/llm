# GPT-style LLM in Python

This project implements a small-scale GPT (Generative Pre-trained Transformer) model from scratch using PyTorch. It includes the model architecture, a training script, and a generation script.

## Project Structure

- `model.py`: Contains the `GPTLanguageModel` architecture, including self-attention, multi-head attention, and transformer blocks.
- `train.py`: Script to train the model on a given text dataset.
- `generate.py`: Script to generate text using a trained model.
- `data/input.txt`: Sample training data (Shakespeare excerpt).
- `model.pth`: Saved model weights (generated after training).

## Requirements

- Python 3.x
- PyTorch

Install dependencies:
```bash
pip install torch
```

## Usage

### 1. Prepare Data
Place your training text in `data/input.txt`.

### 2. Train the Model
Run the training script:
```bash
python train.py
```
This will train the model and save the weights to `model.pth`.

### 3. Generate Text
Run the generation script:
```bash
python generate.py
```

## Model Details

The model is a decoder-only transformer with:
- Multi-head self-attention
- Positional embeddings
- Layer normalization
- Dropout for regularization
- Feed-forward networks

It uses a character-level tokenizer for simplicity.
