import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from reviewlution.utils import clean_for_nlp, custom_stopwords, tokenInit, padding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



class TextProcessor(BaseEstimator, TransformerMixin):
    
    """ Custom Transformer for cleaning and preprocessing string into required format for NN model """
      
    def __init__(self, max_words=5000):
        self.tokenizer = Tokenizer(num_words=max_words)
    
    def fit(self, X, y=None):
        # cleaning text
        X = list(map(clean_for_nlp, X['reviews']))
        self.tokenizer.fit_on_texts(X)
        return self

    def transform(self, X, y=None):
        # cleaning text
        X = list(map(clean_for_nlp, X['reviews']))
        # tokenizing
        sequences = self.tokenizer.texts_to_sequences(X)
        # padding
        X = pad_sequences(sequences, dtype='int32', padding='post')

        return X