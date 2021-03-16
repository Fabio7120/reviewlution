#%tensorflow_version 1.x
import numpy as np
import pandas as pd
from reviewlution.data import get_data, clean_data, balance_data
from reviewlution.encoders import TextProcessor
from reviewlution.utils import  unpack, make_keras_picklable
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from reviewlution.model import initialize_model
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from termcolor import colored
from reviewlution.params import *
import joblib
from google.cloud import storage


class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas DataFrame
        """
        self.pipeline = None
        self.X = X
        self.y = y
   
    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # preprocessing pipeline
        preproc_pipe = Pipeline([('text_preprocessor', TextProcessor())])

        # instantiate pipeline with sklearn wrapper for keras model (to be able to save it into a .joblib format)
        self.pipeline = Pipeline([('preproc', preproc_pipe), ('nn_model', KerasRegressor(build_fn = initialize_model))])


    def run(self):
        """fits pipeline"""
        self.set_pipeline()

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

        self.pipeline.fit(self.X, self.y,
          nn_model__validation_split=0.2,
          nn_model__batch_size=32,
          nn_model__epochs=200,
          nn_model__verbose=1,
          nn_model__callbacks=[es])
        
        print("trained model")


    def evaluate(self):
        """evaluates the pipeline and returns mae"""
        mse = self.pipeline.score(X_test,y_test)
        return mse


    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))
    

    def save_model_to_gcp(self):
        """Save the model into a .joblib and upload it on Google Storage /models folder"""
        local_model_name = 'model.joblib'
        # saving the trained model to disk (which does not really make sense
        # if we are running this code on GCP, because then this file cannot be accessed once the code finished its execution)
        joblib.dump(self.pipeline, local_model_name)
        print("saved model.joblib locally")
        client = storage.Client().bucket(BUCKET_NAME)
        storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
        blob = client.blob(storage_location)
        blob.upload_from_filename(local_model_name)
        print("uploaded model.joblib to gcp cloud storage under \n => {}".format(storage_location))


if __name__ == "__main__":
    # Get and clean data
    N = 520000
    df = get_data(nrows=N)
    df = clean_data(df)
    df = balance_data(df)
    y = df[["review_score"]]
    X = df[['reviews']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Train and save model
    make_keras_picklable()
    trainer = Trainer(X=X_train, y=y_train)
    trainer.run()
    mse = trainer.evaluate()
    print(f"mse: {mse}")
    trainer.save_model_to_gcp()