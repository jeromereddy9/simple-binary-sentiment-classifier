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
        self.vocab = preprocessor.get_vocab()
        self.embedded_vocab = self.build_embedded_vocab()
        self.x_test, self.x_train, self.x_val, self.y_test, self.y_train, self.y_val = self.preprocessor.get_train_test_val_splits()

    def build_embedded_vocab(self):
        size = len(self.vocab)
        embedded_vocab = np.zeros(size,dtype=1)
        for i in range(size):
            if self.vocab[i] == '':
                embedded_vocab[i] = [0] * self.dim
            else:
                embedded_vocab[i] = [r.uniform(-0.1, 0.1) for i in range(self.dim)]

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


    def embed_sequence(self,sequence):
        preprocessed_sequence = self.preprocessor.preprocess_sequence(sequence)
        embedded_sequence = np.zeros(len(preprocessed_sequence),dtype=float)

        for i in range(len(preprocessed_sequence)):
            index = preprocessed_sequence[i]
            embedded_sequence = self.embedded_vocab[index]

        return embedded_sequence


    def average_sequence_vectors(self,embedded_sequence_vectors):
        sum_vector = [sum(embedded_sequence_vectors[i]) for i in range(len(embedded_sequence_vectors))]
        average = sum(sum_vector)/len(sum_vector)
        return average

    def activation(self,average,softmax=True):
        average_tensor = torch.tensor(average)
        if softmax:
            return torch.softmax(average_tensor,1)
        else:
            return torch.sigmoid(average_tensor)

    def classifier(self,activation,threshold=0.5):
        if activation > threshold:
            return 1
        elif activation <= threshold:
            return 0
        return None

    def fit(self,epochs=10,lr=0.1,save_model=True):
        for epoch in range(epochs):
            correct = 0
            for sentence,label in zip(self.x_train,self.y_train):
                embedded_sentence = self.embed_sequence(sentence)
                average = self.average_sequence_vectors(embedded_sentence)
                prediction = self.classifier(self.activation(average))

                if prediction == label:
                    correct += 1
                else:
                    direction = 1 if prediction == 1 else 0
                    for token in sentence:
                        self.embedded_vocab[token] += direction * lr

            training_accuracy = (correct/ len(self.y_train)) * 100
            correct = 0

            for sentence,label in zip(self.x_val,self.y_val):
                embedded_sentence = self.embed_sequence(sentence)
                average = self.average_sequence_vectors(embedded_sentence)
                prediction = self.classifier(self.activation(average))

                if prediction == label:
                    correct += 1

            val_accuracy = (correct / len(self.y_val)) * 100
            print(f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {training_accuracy:.2f}% | Validation Accuracy: {val_accuracy:.2f}%")

        if save_model:
            self.save_model()


    def predict(self,sentence):
        embedded_sentence = self.embed_sequence(sentence)
        average = self.average_sequence_vectors(embedded_sentence)
        prediction = self.classifier(self.activation(average))

        return 'Positive' if prediction == 1 else 'Negative'








        
        
