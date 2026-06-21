import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils import path_builder

def clean_data(data):
    data_lower = tf.strings.lower(data)
    return tf.strings.regex_replace(data_lower,r'[^a-zA-Z0-9\s?!]','')

class Preprocessing:
    def __init__(self,dataset_path,max_sequence_length=12):
        self.dataset_path = dataset_path
        self.data = pd.read_csv(path_builder(dataset_path),encoding='utf-8',dtype='str')
        self.max_sequence_length = max_sequence_length
        self.vectorizer = TextVectorization(
            standardize=clean_data,
            split='whitespace',
            output_mode='int',
            output_sequence_length=self.max_sequence_length
        )
        self.raw_text = self.data.iloc[:,0].values
        self.vectorizer.adapt(self.raw_text)
        self.vocab_dict = self.build_vocab_dict()

    def get_cleaned_data(self):
        return (self.vectorizer(self.raw_text)).numpy()

    def get_vocabulary(self):
        return self.vectorizer.get_vocabulary()

    def build_vocab_dict(self):
        vocab = self.get_vocabulary()
        vocab_dict = {index:word for index,word in enumerate(vocab)}
        return vocab_dict

    def get_vocab_dict(self):
        return self.vocab_dict



path = 'data/IMDB Dataset.csv'
p = Preprocessing(path)








