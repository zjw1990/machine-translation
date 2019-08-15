import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size,hidden_size)
        
        # hidden layer
        self.gru = nn.GRU(hidden_size, hidden_size)
        # output layer
        self.out_layer = nn.Linear(hidden_size, output_size)
        # softmax layer
        self.softmax = nn.LogSoftmax(dim = 1)

    def init_hidden(self):
        return torch.zeros(1,1, self.hidden_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1,1,-1)
        output = F.relu(output)
        # gru
        output, hidden = self.gru(output, hidden)
        # softmax
        output = self.softmax(self.out_layer(output[0]))
        return output, hidden



