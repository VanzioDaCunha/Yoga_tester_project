"""
Created by Vanzio on 18/02/2024
last modified: 18/02/2024 11:23
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed


# Function create an tensor flow model structure which can be called and trained on
# takes input as input shape and give an tensor object as an output
def create_model(input_shape):

    model = Sequential()

    # layers in the model
    #####################################################################################
    model.add(TimeDistributed(Dense(133, activation='tanh'), input_shape=input_shape))

    model.add(Dense(64, activation='tanh'))
    model.add(LSTM(32, activation='tanh', return_sequences=True, return_state=False))

    model.add(Dense(6, activation='softmax'))
    ######################################################################################

    # This line builds the model architecture
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model


sequence_length = 5
num_features = 133
ip_shape = (sequence_length, num_features)
a = create_model(ip_shape)
