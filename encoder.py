import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder():
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        # make embeddings into dim of hidden size
        self.embedding = nn.Embedding(input_size, hidden_size)
        # define GRU
        self.gru = nn.GRU(hidden_size, hidden_size)

    
    def forward(self, input, hidden):
        # embedding size: 1*1*256
        embedding = self.embedding(input).view(1,1,-1)
        output = embedding
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    
    def initLayers(self):
        return torch.zeros(1, 1, self.hidden_size, device = device)





class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)