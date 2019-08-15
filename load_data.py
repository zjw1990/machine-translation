import numpy as np
from preprocess import normalizeString

class Language():
    def __init__(self, name):
        
        self.name = name
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {'SOS':0, 'EOS':1}
        self.voc_size = 2

    def add_word(self, word):
        
        if word not in self.word2idx:
            self.word2idx[word] = self.voc_size
            self.word2count[word] = 1
            self.idx2word[self.voc_size] = word
            self.voc_size += 1

        else:
             self.word2count[word] += 1
    
    def add_sentence(self, sentence):
        
        for word in sentence.split(' '):
            self.add_word(word)



def load_data(path, num):

    with open(path) as f:
        lines = f.readlines()
    pairs =[ [normalizeString(s) for s in l.split('\t')] for l in lines[:num]]

    eng = Language('eng')
    fra = Language('Fra')

    for pair in pairs:
        eng.add_sentence(pair[0])
        fra.add_sentence(pair[1])
    
    return eng, fra, pairs


