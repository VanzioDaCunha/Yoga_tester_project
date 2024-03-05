"""
Created by Vanzio on 18/02/2024
last modified: 18/02/2024 11:23
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from constants import MODEL_INPUT, CLASS_OUTPUT
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy as sparse


# Function create an tensor flow model structure which can be called and trained on
# takes input as input shape and give an tensor object as an output
def create_model(input_shape):

    model = Sequential()

    # layers in the model
    #####################################################################################

    model.add(TimeDistributed(Dense(MODEL_INPUT, activation='relu'), input_shape=input_shape))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    model.add(LSTM(CLASS_OUTPUT, activation='relu', return_sequences=True, return_state=False))
    model.add(Dense(CLASS_OUTPUT, activation='softmax'))

    ######################################################################################

    # This line builds the model architecture
    model.compile(optimizer=Adam(), loss=sparse, metrics=['accuracy'])

    # print(model.summary())

    return model

