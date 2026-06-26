import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.preprocessing import Preprocessor
import pickle as pkl
import numpy as np
import random as r
from src.utils import path_builder
import torch


class Custom_Classifier:
    def __init__(self,preprocessor=None,dim=8):
        self.preprocessor = preprocessor
        self.dim = dim
        self.vocab = preprocessor.get_vocabulary()
        self.embedded_vocab = self.build_embedded_vocab()
        self.x_test, self.x_train, self.x_val, self.y_test, self.y_train, self.y_val = self.preprocessor.get_train_test_val_splits()

    def build_embedded_vocab(self):
        size = len(self.vocab)
        embedded_vocab = []
        for i in range(size):
            if self.vocab[i] == '':
                vector =[0] * self.dim
            else:
                vector = [r.uniform(-0.1, 0.1) for i in range(self.dim)]
            embedded_vocab.append(vector)

        return embedded_vocab

    def load_model(self,path):
        try:
            model = pkl.load(open(path_builder(path), "rb"))
            self.preprocessor = model['Preprocessor']
            self.dim = model['Dimension']
            self.vocab = model['Vocabulary']
            self.embedded_vocab = model['Embedded Vocabulary']
        except FileNotFoundError:
            model = pkl.load(open(path, "rb"))
            self.preprocessor = model['Preprocessor']
            self.dim = model['Dimension']
            self.vocab = model['Vocabulary']
            self.embedded_vocab = model['Embedded Vocabulary']


    def save_model(self,path='src/models/saved_models'):
        model = {'Preprocessor':self.preprocessor,
                 'Dimension':self.dim,
                 'Vocabulary':self.vocab,
                 'Embedded Vocabulary':self.embedded_vocab}
        try:
            pkl.dump(model, open(path_builder(path+'/custom_classifier_model.pkl'), "wb"))
        except FileExistsError:
            pkl.dump(model, open(path+'/custom_classifier_model.pkl', "wb"))


    def embed_sequence(self,sequence,preprocess=False):
        if preprocess:
            preprocessed_sequence = self.preprocessor.preprocess_sequence(sequence)
            embedded_sequence = []
            for i in range(len(preprocessed_sequence)):
                index = preprocessed_sequence[i]
                embedded_sequence.append(self.embedded_vocab[index])
        else:
            embedded_sequence = []
            for i in range(len(sequence)):
                index = sequence[i]
                embedded_sequence.append(self.embedded_vocab[index])

        return embedded_sequence


    def average(self,embedded_sequence_vectors):
        pass

    def activation(self,embedded_sequences):
        pass

    def classifier(self,activation,threshold=0.5):
        pass

    def fit(self,epochs=10,lr=0.025,save_model=True):
       pass


    def predict(self,sentence):
        pass



path = 'data/IMDB Dataset.csv'
p = Preprocessor(path)

model = Custom_Classifier(preprocessor=p)

model.fit(5)




        
        
