import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BatchDecoder(nn.Module):
    def __init__(self, voc_size_source, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(BatchDecoder, self).__init__()
        # parameters

        self.voc_size_source = voc_size_source
        self.hidden_size = hidden_size
        # voc_size_target
        self.output_size = output_size
        self.dropout = dropout
        self.n_layers=  n_layers
        # layers
        self.embedding = nn.Embedding(voc_size_source, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = 