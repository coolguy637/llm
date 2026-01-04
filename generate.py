import torch
from model import GPTLanguageModel

# Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data to get vocab_size and tokenizer
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Load the model
model = GPTLanguageModel(vocab_size)
try:
    model.load_state_dict(torch.load('model.pth', map_location=device))
    print("Loaded pre-trained model.")
except FileNotFoundError:
    print("No pre-trained model found. Please run train.py first.")
    exit()

model.to(device)
model.eval()

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device) # start with a newline character (index 0)
print("Generated text:")
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
