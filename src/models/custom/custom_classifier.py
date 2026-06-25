import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.preprocessing import Preprocessor
import pickle as pkl
import numpy as np
import random as r
from src.utils import path_builder


class Custom_Classifier:
    def __init__(self,preprocessor,dim=8):
        self.preprocessor = preprocessor
        self.dim = dim
        self.vocab = preprocessor.get_vocab()
        self.embedded_vocab = self.build_embedded_vocab()

    def build_embedded_vocab(self):
        size = len(self.vocab)
        embedded_vocab = np.zeros(size,dtype=1)
        for i in range(size):
            if self.vocab[i] == '':
                embedded_vocab[i] = [0] * self.dim
            else:
                embedded_vocab[i] = [r.uniform(-0.1, 0.1) for i in range(self.dim)]

        return embedded_vocab

    def load_embedded_vocab(self,path):
        try:
            self.embedded_vocab = pkl.load(open(path_builder(path), "rb"))
        finally:
            self.embedded_vocab = pkl.load(open(path, "rb"))


    def save_embedded_vocab(self,path):
        try:
            pkl.dump(self.embedded_vocab, open(path_builder(path), "wb"))
        finally:
            pkl.dump(self.embedded_vocab, open(path, "wb"))


    def embed_sequence(self,sequence):
        pass


    def average_sequence_vectors(self,sequence_vectors):
        pass



        
        
