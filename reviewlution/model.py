
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers

def initialize_model():
    """NN Model Architecture"""
    ### Model
    model = models.Sequential()
 
    ### Embedding Padded
    model.add(layers.Embedding(input_dim=5000, output_dim=100, mask_zero=True))
        
    ### First convolution & max-pooling
    model.add(layers.LSTM(units=100, activation='tanh', return_sequences=True))
    model.add(layers.LSTM(units=100, activation='tanh', return_sequences=True))
    model.add(layers.LSTM(units=50, activation='tanh'))
    model.add(layers.Dropout(0.2))                     #change params
    model.add(layers.Dense(40, activation='relu', kernel_regularizer=regularizers.L1(0.01)))    #Use regulazers
    model.add(layers.Dropout(0.2))                     #change params
    model.add(layers.Dense(20, activation='relu', kernel_regularizer=regularizers.L1(0.01)))    #Use regulazers
    model.add(layers.Dropout(0.2))                     #change params
    model.add(layers.Dense(10, activation='relu', kernel_regularizer=regularizers.L1(0.01)))    #Use regulazers
    model.add(layers.Dropout(0.2))                     #change params 

    ### Last layer (let's say a classification with 10 output)
    model.add(layers.Dense(1, activation='linear'))
        
    ### Model compilation
    model.compile(loss='mse', 
                  optimizer='rmsprop',
                  metrics=['mae'])     

    return model