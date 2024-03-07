"""
Created by Divya on 18/02/2024
last modified: 18/02/2024 11:23
"""


import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from constants import MODEL_INPUT, LABELS
from constants import OUTPUT_FILE_PATH


# Function takes csv file as an input and returns 2 Numpy arrays
def data_preprocessing(filename):
    filename = OUTPUT_FILE_PATH + filename
    arr = []
    # Using load txt() to write data from csv file to arr list
    arr = np.loadtxt(filename, delimiter=",", dtype=str)

    # X_list to store current time along with keypoints
    # Y_list to store the activity corresponding to the current time
    X_list = []
    Y_list = []
    for data in arr:
        line = []
        flag = 0
        for i, value in enumerate(data):
            if value == '':
                flag = 1
                break
            elif i == MODEL_INPUT:
                Y_list.append(value)
            else:
                line.append(float(value))
        if flag == 0:
            X_list.append(line)

    # converting list to Numpy array
    x = np.array(X_list)
    y = np.array(Y_list)
    z = np.unique(y)

    # Use LabelEncoder to convert string labels to integers
    label_encoder = LabelEncoder()
    label_encoder.fit(LABELS)

    y = label_encoder.transform(y)
    y = to_categorical(y)
    return x, y
