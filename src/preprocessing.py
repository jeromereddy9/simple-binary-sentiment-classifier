import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils import path_builder
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from sklearn.model_selection import train_test_split
import numpy as np

def clean_data(data):
    data_lower = tf.strings.lower(data)
    return tf.strings.regex_replace(data_lower,r'[^a-zA-Z0-9\s?!]','')

class Preprocessor:
    def __init__(self,dataset_path=None,max_sequence_length=8,training_split=0.7,val_split=0.15,test_split=0.15):
        self.dataset_path = dataset_path
        self.training_size = training_split
        self.test_size = test_split
        self.val_size = val_split
        self.data = pd.read_csv(path_builder(dataset_path),encoding='utf-8',dtype='str')
        self.max_sequence_length = max_sequence_length
        self.vectorizer = TextVectorization(
            standardize=clean_data,
            split='whitespace',
            output_mode='int',
            output_sequence_length=self.max_sequence_length
        )
        self.raw_text = self.data.iloc[:,0].values
        self.labels = self.data.iloc[:,1].values
        self.vectorizer.adapt(self.raw_text)

    def get_cleaned_data(self):
        return (self.vectorizer(self.raw_text)).numpy()

    def get_vocabulary(self):
        return self.vectorizer.get_vocabulary()

    def preprocess_sequence(self,sequence):
        return (self.vectorizer(sequence)).numpy()

    def get_train_test_val_splits(self):
        np.random.seed(42)
        x = self.get_cleaned_data()
        y = self.labels
        x_temp,x_test,y_temp,y_test = train_test_split(x,y,test_size=self.test_size,random_state=42,stratify=y)
        x_train,x_val,y_train,y_val = train_test_split(x_temp,y_temp,test_size=self.val_size,random_state=42,stratify=y_temp)

        return x_test,x_train,x_val,y_test,y_train,y_val
















