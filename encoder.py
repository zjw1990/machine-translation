import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    # input_size: vocabulary size
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        # project from original dimension to hidden_size dimension 1 -> 256
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
    
    def init_hidden(self):
        return torch.zeros(1,1, self.hidden_size)
    

    def forward(self, input, hidden):
        # input here is a single index of a word in a specific sentence e.g. [9]
        # embedding size 1*1*hidden_size
        embedded = self.embedding(input).view(1,1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)

        return output, hidden