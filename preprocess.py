import re
import unicodedata
import torch
import numpy as np
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 30
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# transfer words in languages into indexes
def indexesFromSentence(lang, sentence):
    senence_indexes = [lang.word2idx[word] for word in sentence.split(' ')] # a list with length of input
    return senence_indexes

# transfer the index into a tensor, ,padding, add a eos token
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    padded_indexes = np.zeros(MAX_LENGTH)
    for i in range(len(indexes)):
        padded_indexes[i] = indexes[i]
    sentence_tensor = torch.tensor(padded_indexes, dtype=torch.long, device = device).view(-1, 1)
    return sentence_tensor


def tensorsFromPair(pair, input_lang, output_lang):
    
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)