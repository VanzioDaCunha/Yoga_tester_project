"""
Created by Vanzio on 18/02/2024
last modified: 09/04/2024 13:16
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Reshape
from constants import CLASS_OUTPUT, SEQUENCE_LENGTH, MODEL_INPUT
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy as sparse


# Function create an tensor flow model structure which can be called and trained on
# takes input as input shape and give an tensor object as an output
def create_model(input_shape):

    model = Sequential()

    # layers in the model
    #####################################################################################

    model.add(Reshape((SEQUENCE_LENGTH, MODEL_INPUT, 1), input_shape=input_shape))
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dense(units=CLASS_OUTPUT, activation='softmax'))

    ######################################################################################

    # This line builds the model architecture
    model.compile(optimizer=Adam(), loss=sparse, metrics=['accuracy'])

    # print(model.summary())

    return model

