import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch import LongTensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class BatchEncoder(nn.Module):
    # inital model components
    def __init__(self, voc_size, embedding_size, hidden_size, n_layers = 1, dropout = 0.5):
        super(BatchEncoder, self).__init__()
        # input: vocabulary size, for nn.Embedding look_up table.
        self.voc_size = voc_size
        # word embedding dim
        self.embedding_size = embedding_size
        # hidden layer size
        self.hidden_size = hidden_size
        # hidden layer number
        self.n_layers = n_layers
        # dropout prob
        self.dropout = dropout
        
        # inital embedding model
        self.embedding = nn.Embedding(voc_size, embedding_size)
        # inital hidden model
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout = dropout)

    def forward(self, input_seqs, hidden=None):
        # B: batch_size
        # L: sequence lengths for all sentences in batches (already padded)
        
        # input_seqs : a mini-batch input, with a shape of [B, L]
        # input_seqs => [[ 6  9  8  4  1 11 12 10]          # long_str
        #                 [12  5  8 14 ]                    # tiny
        #                 [ 7  3  2  5 13  7 ]]             # medium
        
        # seq_lengths: a list that contains all real length of input sentences.
        # seq_lengths => [8, 4, 6]
        ######                      step 1 preprocess input               #################
        
        
        ### 1.1 get seq length of all sentences
        
        # seq_lengths: a list that contains all real length of input sentences.
        # seq_lengths => [8, 4, 6]
        seq_lengths = LongTensor(list(map(len, input_seqs)))
        
        
        ### 1.2 padding
        
        # inital input_seqs_tensor => [[0 0 0 0 0 0 0 0]
        #                               [0 0 0 0 0 0 0 0]
        #                               [0 0 0 0 0 0 0 0]]
        input_seqs_tensor = Variable(torch.zeros((len(input_seqs), seq_lengths.max()))).long()
        
        # padding:   [[ 6  9  8  4  1 11 12 10]          # long_str
        #              [12  5  8 14  0  0  0  0]         # tiny
        #              [ 7  3  2  5 13  7  0  0]]        # medium
        # input_seqs_tensor.shape : (batch_size X max_seq_len) = (3 X 8)
        
        for idx, (seq, seqlen) in enumerate(zip(input_seqs, seq_lengths)):
             input_seqs_tensor[idx, :seqlen] = LongTensor(seq)

       
        ### 1.3 sort seqs by length in a decending order
        
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        input_seqs_tensor = input_seqs_tensor[perm_idx]
        # input_seqs => [[ 6  9  8  4  1 11 12 10]           # long_str
        #                [ 7  3  2  5 13  7  0  0]           # medium
        #                [12  5  8 14  0  0  0  0]]          # tiny
        
        
        ######                      step 2 forward               #################
        
        ### 2.1  inital embeddings, shape [B,L,D]
        embedings = self.embedding(input_seqs_tensor)

        ### 2.2 pack embeddings for batch-training
        packed_embeddings = pack_padded_sequence(embedings, seq_lengths.cpu().numpy(), batch_first=True)
        # packed_embeddings.data.shape : (batch_sum_seq_len X embedding_dim) = (18 X D)
        # packed_embeddings.batch_sizes => [ 3,  3,  3,  3,  2,  2,  1,  1] 
        # visualization :               => | l | o | n | g | _ | s | t | r |  #(long_str)
        #                                  | m | e | d | i | u | m |   |   |  #(medium)
        #                                  | t | i | n | y |   |   |   |   |  #(tiny)
        
        ### 2.3 training with GRU 
        # packed_output.data.shape : (batch_sum_seq_len X hidden_dim) = (18 X 5)
        # packed_output.batch_sizes => [ 3,  3,  3,  3,  2,  2,  1,  1] (same as packed_input.batch_sizes)
        outputs_packed, hidden = self.gru(packed_embeddings, hidden)


        
        
        hidden = pad_packed_sequence(hidden)
        
        return outputs_packed, hidden