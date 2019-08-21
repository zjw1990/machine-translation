import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        # 256
        self.hidden_size = hidden_size
        # voc_t * 256
        self.embedding = nn.Embedding(output_size,hidden_size)
        
        # hidden layer
        self.gru = nn.GRU(hidden_size, hidden_size)
        # output layer
        self.out_layer = nn.Linear(hidden_size, output_size)
        # softmax layer
        self.softmax = nn.LogSoftmax(dim = 1)

    def init_hidden(self):
        return torch.zeros(1,1, self.hidden_size, device = device)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1,1,-1)
        output = F.relu(output)
        # gru
        output, hidden = self.gru(output, hidden)
        # softmax
        output = self.softmax(self.out_layer(output[0]))
        return output, hidden



class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p):
        super(AttentionDecoder, self).__init__()
        
        # 256
        self.hidden_size =  hidden_size
        # target voc_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        # input sentence length
        
        # structure
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        # Linear model, input: concanation of s(i-1) and h(j) output: eij
        # I: len(input seq)  dim(s):256
        # J: len(output seq) dim(h):256
        self.align_model = nn.Linear(self.hidden_size*2, 1)
        # calculate current hidden state s(i)
        self.new_state = nn.Linear(self.hidden_size*2, self.hidden_size)
        
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device = device)

    
    def forward(self, y_before, s_before, h_all):
        # input:
        ## y_before -- shape 1*1*1 : previous output  y(i-1) 
        ## s_before -- shape 1*1*256 : previous hidden state s(i-1)
        ## h_all 
        # -- shape input_sentence_length*256
        # all encoder's hidden states h_all


        # Step1 initial input
        # inital a single word into a embedding y(i-1)
        # embedding size(1,1,256) 
        y_before_emb = self.embedding(y_before).view(1, 1, -1)
        y_before_emb = self.dropout(y_before_emb)
        # Concatenates the given sequence of seq tensors in the given dimension. 
        # All tensors must either have the same shape (except in the concatenating dimension) or be empty.
        h_all = h_all.unsqueeze(0) # 1*10*256
        s_before_ex = s_before.expand(1, len(h_all[0]), -1) # 1*10*256
        # align model, calculate e
        align_model_input = torch.cat((h_all, s_before_ex), dim = 2) # 1*10*512
        e = self.align_model(align_model_input).view(1,1,-1) # 1*1*10
        # weight
        a = F.softmax(e, dim = 2) # 1*1*10
        # context vector
        # Returns a new tensor with a dimension of size one inserted at the specified position. 
        # The returned tensor shares the same underlying data with this tensor.
        c = torch.bmm(a, h_all) # 1*1*256

        output = torch.cat((y_before_emb[0], c[0]), 1) # 1*512
        output = self.new_state(output).unsqueeze(0) # 1*1*256

        output, s_now = self.gru(output, s_before)
        
        output = F.log_softmax(self.out(output[0]), dim=1) # output 1*3050

        return output, s_now, a[0]
    



