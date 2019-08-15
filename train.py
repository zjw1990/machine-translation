import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_data import load_data
from preprocess import tensorFromSentence, tensorsFromPair, indexesFromSentence
from encoder import Encoder
from decoder import Decoder
import time
import random
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# global parameters
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SOS_token = 0
EOS_token = 1
START_TIME = time.time()

# training parameters
epochs = 10000
learning_rate = 0.03
hidden_size = 256

# load data
input_lang,output_lang, pairs = load_data('./data/eng-fra.txt', 1000)
print('successfully loaded data')

# start to train

plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every

# initial model 
encoder = Encoder(input_lang.voc_size, hidden_size)
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)

decoder = Decoder(hidden_size, output_lang.voc_size)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

# inital input, random choose a sentence from all data.
training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang) for i in range(epochs)]
loss_function = nn.NLLLoss()

for epoch in range(1, epochs+1):
    training_pair = training_pairs[epoch - 1]
    # sentence*1
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]
    
    # Build Encoder
    
    ## initial encoder rnn hidden layer
    encoder_hidden = encoder.init_hidden()
    ## initial encoder grads
    encoder_optimizer.zero_grad()
    ## encoder input_size, target_size
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    ## build a tensor to contain all outputs for a sentence
    encoder_outputs = torch.zeros(target_length, encoder.hidden_size)
    ## initial loss
    loss = 0
    ## start put data into encoder rnn, forward
    for ei in range(input_length):
        # forwad for a single word
        encoder_output, encoder_hidden = encoder.forward(input_tensor[ei], encoder_hidden) 
        #encoder_outputs[ei] = encoder_output[0, 0]
    
    
    # Decoder
    
    ## initial input, hidden of decoder rnn
    decoder_input = torch.tensor([[SOS_token]])
    decoder_hidden = encoder_hidden
    ## initial dencoder grads
    decoder_optimizer.zero_grad()
    ## training decoder rnn
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder.forward(decoder_input, encoder_hidden)
        decoder_input =  target_tensor[di]  
        loss += loss_function(decoder_output, target_tensor[di])
        #print(decoder_input.item())
        if decoder_input.item() == EOS_token:
            break
    
    # Training end start to BP
    loss.backward()
    # Clear the gradients in both encoder and decoder
    encoder_optimizer.step()
    decoder_optimizer.step()

    loss_iter  = loss.item() / target_length
    
    print_loss_total += loss_iter
    plot_loss_total += loss_iter
    print_every = 1000
    
    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print('%s (%d %d%%) %.4f' % (timeSince(START_TIME, epoch / epochs), epoch, epoch / epochs * 100, print_loss_avg))



# evaluate
# training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang) for i in range(3)]
training_pairs = []
for i in range(3):
    sentences = random.choice(pairs)
    print(sentences)
    training_pairs.append(tensorsFromPair(sentences, input_lang, output_lang))

for sentence in training_pairs:

    # output_words, attentions = evaluate(encoder, decoder, pair[0])

    input_tensor = sentence[0]
    target_tensor = sentence[1]

    with torch.no_grad():
        # Read the sentence from the testing set. 
 
        # Initial 
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_hidden = encoder.init_hidden()
        # Encoder outputs
        encoder_outputs = torch.zeros(target_length, encoder.hidden_size)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        
        decoder_input = torch.tensor([[SOS_token]])  # SOS
        decoder_hidden = encoder_hidden
        
        decoded_words = []
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder.forward(decoder_input, decoder_hidden)
            value, indice = decoder_output.data.topk(1)
            decoder_input = indice
            if indice.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.idx2word[indice.item()])


    output_sentence = ' '.join(decoded_words)
    print('output', output_sentence)
    print('')
