import torch
import torch.nn as nn
from torch.nn import functional as F

import sys


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# read it in to inspect it
with open('data/data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# define encoding and decoding functions (tokenizer)
chars = sorted(list(set(text)))
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]          # encoder: take a string and output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# encoding the entire dataset into a tensor
data = torch.tensor(encode(text), dtype=torch.long)

# build dataset
n = int(0.9 * len(data)) 
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

def get_num_params():
    # print the number of parameters of the model
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{(n_params / 1e6):.6f} M learnable parameters\n')

@torch.no_grad()
def estimate_loss(eval_iters):
    # estimate the loss. the higher eval_iters is, the more precise the loss will be
    print('estimating loss...')
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    print(f'estimated loss: {out}\n')

@torch.no_grad()
def generate(max_new_tokens):
    # generate text (output from the model)
    idx = torch.tensor([0]).to(device) # seed for generating text (aka starting character)
    print('generating new text...')
    for i in range(max_new_tokens):
        logits, loss = model(idx, y=None, infering=True)
        logits = logits[-1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next))
    print(decode(idx.tolist()))

def train(max_steps):
    # train the model
    print("training...")
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for steps in range(max_steps + 1):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if steps % (0.1 * max_steps) == 0:
            print(f'loss {loss.item():.6f} | {steps/max_steps * 100:.2f}%')
    print('training completed\n')

            
# ------------------------------ Long Short-Term Memory -----------------------------------------------
class LSTM(nn.Module): 
    
    def __init__(self, hidden_size, output_size, embd_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.emb = nn.Embedding(output_size, embd_size) 
        self.linear = nn.Linear(embd_size, hidden_size)
        self.forget_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )
        self.input_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh()
            ),
        ])
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )
        self.tanh = nn.Tanh()
        self.lm_head = nn.Linear(hidden_size, output_size)

    def forward(self, X, y=None, infering=False):
        # embed the characters
        X = self.emb(X)
    
        # reshape the embeddings
        X = self.linear(X) # (hidden_size, 1)

        # takes ht-1 and ct-1 and outputs h and c.
        def lstm_cell(c, h, x):

            # concatenate input to h
            h_x = torch.concat((h, x), dim=-1) # (hidden_size +  hidden_size, 1)

            
            # forget gate
            fout = self.forget_gate(h_x) # (hidden_size, 1)
            # update c
            c = c * fout

            # input gate
            isig = self.input_gate[0](h_x) # (hidden_size, 1)
            itanh = self.input_gate[1](h_x) # (hidden_size, 1)
            iout = isig * itanh 
            # update c
            c = c + iout 

            # output gate    
            oout = self.output_gate(h_x) # (hidden_size, 1)
            h = self.tanh(c) * oout
            
            return c, h # (hidden_size, 1) both

        if not infering:
            B, T, C = X.shape
            # initialize h and c to 0
            c = torch.zeros(B, C).to(device)
            h = torch.zeros(B, C).to(device)

            to_fill = torch.zeros((B, T, self.hidden_size)).to(device)
        
            for i in range(T):
                x = X[:, i]
                #print(x.shape)
                c, h = lstm_cell(c, h, x)
                to_fill[:, i] = h
        else:
            T, C = X.shape
            # initialize h and c to 0
            c = torch.zeros(C).to(device)
            h = torch.zeros(C).to(device)

            to_fill = torch.zeros((T, C)).to(device)
    
            for i in range(T):
                x = X[i]
                c, h = lstm_cell(c, h, x)
                to_fill[i] = h

        # get logits
        logits = self.lm_head(to_fill) * 0.1

        if y is None:
             return logits, None
            
        # calculate loss
        loss = F.cross_entropy(logits.view(batch_size * block_size, self.output_size), y.view(batch_size * block_size))
        return logits, loss

    
# Hyperparameters-------
vocab_size = len(chars)
hidden_size = 512
embd_size = 192
batch_size = 32
block_size = 512
learning_rate = 3e-4
# ----------------------

# Initialize model
model = LSTM(hidden_size=hidden_size, output_size=vocab_size, embd_size=embd_size)
model = model.to(device)
        
def main():

    # training, evaluation and generation parameters
    training_steps = 50000
    evaluation_steps = 500
    new_tokens = 500

    # Check for loading pretrained model
    if len(sys.argv) != 1:
        try:
            model.load_state_dict(torch.load(sys.argv[1])) 
            get_num_params()
            generate(new_tokens)
        except:
            sys.exit('could not load pre trained model')
    else:
        get_num_params()
        train(training_steps)
        estimate_loss(evaluation_steps)
        generate(new_tokens)


if __name__ == '__main__':
    main()