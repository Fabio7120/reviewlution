import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from reviewlution.params import *
from reviewlution.utils import *

def get_data(nrows=520_000):
    '''returns a DataFrame with nrows from downloaded Keggle csv in raw_data folder'''
    dataset_1 = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}", nrows=nrows)
    df = dataset_1.copy()
    return df


def clean_data(df):
    '''returns cleaned DataFrame'''
    
    # dropping redundant columns
    df = df[['Negative_Review', 'Positive_Review', 'Reviewer_Score']]

    # Cleaning, merging and renaming negative and positive reviews
    df[['Negative_Review']] = df[['Negative_Review']].replace(to_replace="No Negative", value="")
    df[['Positive_Review']] = df[['Positive_Review']].replace(to_replace="No Positive", value="")
    df["reviews"] = df['Negative_Review'] + " " + df['Positive_Review']
    df["review_score"] = df['Reviewer_Score']
    df = df.drop(columns=['Negative_Review', 'Positive_Review', 'Reviewer_Score'])

    # Clean for nlp
    clean_primary_data(df)

    # Remove reviews with less than 6 words (or signs)
    df['length'] = df['reviews'].apply(lambda x: len(word_tokenize(str(x))))
    df.drop(df[df['length'] < 6].index, inplace=True)
    df.drop(columns=['length'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def balance_data(df):
    '''returns balanced DataFrame'''
    
    df_1 = df[df['review_score'] < 5][:10000]
    df_4 = df[(df['review_score'] > 9) & (df['review_score'] < 10.1)][:10000]
    df = pd.concat([df_1,df_4])
    df.reset_index(drop=True, inplace=True)

    return df