import torch
import torch.nn.functional as F
import numpy as np

def load_data():
    #TODO: download dataset and finish load data and corresponding part in train.py
    pass

def create_pad_mask(t, pad):
    mask = (t == pad).unsqueeze(-2)
    return mask


def create_trg_self_mask(target_len, device=None):
    # Prevent leftward information flow in self-attention.
    ones = torch.ones(target_len, target_len, dtype=torch.uint8,
                      device=device)
    t_self_mask = torch.triu(ones, diagonal=1).unsqueeze(0)

    return t_self_mask