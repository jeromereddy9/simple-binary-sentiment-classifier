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
import torch.nn as nn


class Custom_Classifier:
    def __init__(self,preprocessor=None,dim=8,lr = 0.025):
        self.preprocessor = preprocessor
        self.dim = dim
        self.lr = lr
        self.epochs = 10
        if self.preprocessor is not None:
            self.vocab = preprocessor.get_vocabulary()
            self.embedded_vocab = self.build_embedded_vocab()
            self.x_test, self.x_train, self.x_val, self.y_test, self.y_train, self.y_val = self.preprocessor.get_train_test_val_splits()
        else:
            self.vocab = None
            self.embedded_vocab = None
            self.x_test, self.x_train, self.x_val, self.y_test, self.y_train, self.y_val = None,None,None,None,None,None

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

    def load_model(self,path='src/models/saved_models',name='/custom_classifier_model.pkl'):
        try:
            model = pkl.load(open(path_builder(path+name), "rb"))
            self.preprocessor = model['Preprocessor']
            self.dim = model['Dimension']
            self.vocab = model['Vocabulary']
            self.embedded_vocab = model['Embedded Vocabulary']
            self.lr = model['Learning Rate']
            self.epochs = model['Epochs']
            self.x_test, self.x_train, self.x_val, self.y_test, self.y_train, self.y_val = self.preprocessor.get_train_test_val_splits()

        except FileNotFoundError:
            model = pkl.load(open(path+name, "rb"))
            self.preprocessor = model['Preprocessor']
            self.dim = model['Dimension']
            self.vocab = model['Vocabulary']
            self.embedded_vocab = model['Embedded Vocabulary']
            self.lr = model['Learning Rate']
            self.epochs = model['Epochs']
            self.x_test, self.x_train, self.x_val, self.y_test, self.y_train, self.y_val = self.preprocessor.get_train_test_val_splits()



    def save_model(self,path='src/models/saved_models',name='/custom_classifier_model.pkl'):
        model = {'Preprocessor':self.preprocessor,
                 'Dimension':self.dim,
                 'Vocabulary':self.vocab,
                 'Embedded Vocabulary':self.embedded_vocab,
                 'Learning Rate':self.lr,
                 'Epochs':self.epochs}
        try:
            pkl.dump(model, open(path_builder(path+name), "wb"))
        except FileExistsError:
            pkl.dump(model, open(path+name, "wb"))


    def embed_sentence(self,sequence,preprocess=False):
        embedded_sentence = []
        if preprocess:
            preprocessed_sequence = self.preprocessor.preprocess_sequence(sequence)
            for i in range(len(preprocessed_sequence)):
                index = preprocessed_sequence[i]
                embedded_sentence.append(self.embedded_vocab[index])
        else:
            for i in range(len(sequence)):
                index = sequence[i]
                embedded_sentence.append(self.embedded_vocab[index])

        return embedded_sentence

    def set_learning_rate(self,lr):
        self.lr = lr

    def get_test_split(self):
        return self.x_test,self.y_test

    def mean_pool(self,embedded_sentence):
        sentence_embedding = np.mean(embedded_sentence,axis=0)
        return sentence_embedding

    def activation(self,sentence_embedding):
        sentence_tensor = torch.tensor(sentence_embedding)
        probability = torch.sigmoid(sentence_tensor)
        return probability

    def calculate_loss(self,y_prediction,y_true):
        prediction = y_prediction.detach().float().flatten()
        true_label = torch.tensor(y_true,dtype=torch.float32).flatten()
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(prediction,true_label)
        return loss

    def threshold_classification(self,probability,threshold=0.5):
        if probability > threshold:
            return 'positive'
        else:
            return 'negative'

    def fit(self,epochs=10,save_model=True,path=None,name=None):
        self.epochs = epochs
        num_samples_train = len(self.y_train)
        num_samples_val = len(self.y_val)

        print("Training Starting:")
        for epoch in range(epochs):
            correct = 0
            train_loss = 0
            for sentence,label in zip(self.x_train,self.y_train):
                embedded_sentence = self.embed_sentence(sentence)
                averaged_sentence = self.mean_pool(embedded_sentence)
                activation = self.activation(averaged_sentence)
                probability = torch.exp(torch.mean(torch.log(activation)))
                classification = self.threshold_classification(probability)

                if classification == label:
                    correct += 1
                label_numerical = 1 if label == 'positive' else 0
                loss = self.calculate_loss(probability,label_numerical)
                train_loss += loss
                error = probability - label_numerical

                gradient = (error * activation * (1-activation))/len(sentence)
                gradient = gradient.detach().numpy()

                for i in range(len(sentence)):
                    token_id = sentence[i]

                    if token_id == 0:
                        continue

                    self.embedded_vocab[token_id] -= self.lr * gradient

            training_accuracy = (correct/num_samples_train) * 100

            correct = 0
            val_loss = 0
            for sentence,label in zip(self.x_val,self.y_val):
                embedded_sentence = self.embed_sentence(sentence)
                averaged_sentence = self.mean_pool(embedded_sentence)
                activation = self.activation(averaged_sentence)
                probability = torch.exp(torch.mean(torch.log(activation)))
                classification = self.threshold_classification(probability)

                if classification == label:
                    correct += 1

                label_numerical = 1 if label == 'positive' else 0
                loss = self.calculate_loss(probability, label_numerical)
                val_loss += loss

            val_accuracy = (correct/num_samples_val) * 100

            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Acc: {training_accuracy:.2f}% | Train Loss: {train_loss/num_samples_train:.4f} | "
                  f"Val Acc: {val_accuracy:.2f}% | Val Loss: {val_loss/num_samples_val:.4f}")

        if save_model:
            if path is None:
                if name is None:
                    self.save_model()
                else:
                    self.save_model(name=name)
            else:
                if name is None:
                    self.save_model(path=path)
                else:
                    self.save_model(path=path,name=name)

    def predict(self,sentence,preprocess=False):
        embedded_sentence = self.embed_sentence(sentence,preprocess)
        averaged_sentence = self.mean_pool(embedded_sentence)
        activation = self.activation(averaged_sentence)
        probability = torch.exp(torch.mean(torch.log(activation)))
        classification = self.threshold_classification(probability)

        return classification

    def get_model_summary(self):
        print("Model Summary:")
        print(f"Dims -> {self.dim}")
        print(f"Learning Rate -> {self.lr}")
        print(f"Max Sequence Length -> {self.preprocessor.get_max_sequence_length()}")
        print(f"Epochs -> {self.epochs}")
















        
        
