"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'enwik8')
if not os.path.exists(input_file_path):
    raise ValueError("enwik8 not found")

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
#print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

def create_mask(l, true_ids):
    return [i in true_ids for i in l]

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):int(n*0.95)]
test_data = data[int(n*0.95):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")
# export to bin files


train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
test_ids.tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

alphabet_ids = encode("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
train_mask = create_mask(train_ids, alphabet_ids)
val_mask = create_mask(val_ids, alphabet_ids)
test_mask = create_mask(test_ids, alphabet_ids)

train_mask = np.array(train_mask, dtype=np.uint8)
val_mask = np.array(val_mask, dtype=np.uint8)
test_mask = np.array(test_mask, dtype=np.uint8)

train_mask.tofile(os.path.join(os.path.dirname(__file__), 'train_mask.bin'))
val_mask.tofile(os.path.join(os.path.dirname(__file__), 'val_mask.bin'))
test_mask.tofile(os.path.join(os.path.dirname(__file__), 'test_mask.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
