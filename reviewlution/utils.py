import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils


def custom_stopwords():
    """create custom stopwords list excluding negative words"""
    negative_words = ['no',
    'nor',
    'not',
    "don't",
    'should',
    "should've",
    'aren',
    "aren't",
    'couldn',
    "couldn't",
    'didn',
    "didn't",
    'doesn',
    "doesn't",
    'hadn',
    "hadn't",
    'hasn',
    "hasn't",
    'haven',
    "haven't",
    'isn',
    "isn't",
    "wasn't",
    'weren',
    "weren't",
    'won',
    "won't",
    'wouldn',
    "wouldn't"]

    custom_stopwords = [x for x in stopwords.words('english') if x not in negative_words]

    #extra_stopwords = ["hotel","everything","anything","thing"]  #customize extra stop_words

    #custom_stopwords.extend(extra_stopwords)

    return custom_stopwords


def clean_for_nlp(text):
    """ preprocess review text data for nlp analysis """
    # Lower case
    text = ''.join(text)
    text = text.lower()
    # Remove numbers
    text = ''.join(word for word in text if not word.isdigit())
    # Remove punctuation
    for punctuation in string.punctuation:
        text = text.replace(punctuation.replace("'","").replace("`",""), ' ')
    # Remove stopwords
    text = word_tokenize(text)
    stopwords = custom_stopwords()
    text = [w for w in text if not w in stopwords]
    # Lemmatizing
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(word for word in text)

    return(text)


def remove_numbers(text):
    """Remove numbers - Input for clean_primary_data (1/2)"""
    text = ''.join(word for word in text if not word.isdigit())

    return text


def lemmatizing(text):
    """Lemmatizing - Input for clean_primary_data (2/2)"""
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in text]
    text = lemmatized
    text = ' '.join(word for word in text)

    return text


def clean_primary_data(df):
    """"clean raw data only"""
    df['reviews'] = df['reviews'].apply(lambda x: x.lower())
    df['reviews'] = df['reviews'].apply(remove_numbers)
    stop_words = custom_stopwords()
    df['reviews'] = df['reviews'].map(word_tokenize)
    df['reviews'] = df['reviews'].map(lambda x: [w for w in x if not w in stop_words])
    for punctuation in string.punctuation:
      df['reviews'] = df['reviews'].replace(string.punctuation.replace("'","").replace("`",""), ' ')
    df['reviews'] = df['reviews'].apply(lemmatizing)

    return df



def tokenInit(train, max_words=5000):
    "Returns tokenized sentences to transform X (for DL model)"
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train)

    return tokenizer


def padding(X):
    "Pads tokenized sequences to transform X (for DL model)"
    tokenizer = tokenInit(X)
    sequences = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(sequences, dtype='int32', padding='post')

    return X_pad


def unpack(model, training_config, weights):
    """Make Keras Model exportable within pipeline ("picklable") 1/2"""
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model


def make_keras_picklable():
     """Make Keras Model exportable within pipeline ("picklable") 2/2"""
     
     def __reduce__(self):
         model_metadata = saving_utils.model_metadata(self)
         training_config = model_metadata.get("training_config", None)
         model = serialize(self)
         weights = self.get_weights()
         return (unpack, (model, training_config, weights))
    
     cls = Model
     cls.__reduce__ = __reduce__
