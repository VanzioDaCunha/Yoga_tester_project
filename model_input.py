"""
Created by Divya on 18/02/2024
last modified: 18/02/2024 11:23
"""


import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from constants import MODEL_INPUT, LABELS, CLASS_OUTPUT
from constants import OUTPUT_FILE_PATH
from typing import Tuple


def data_preprocessing(filename) -> Tuple[np.ndarray, np.ndarray]:
    """
        This Function Takes the csv file and returns the keypoints and it corresponding class

        Args:
            filename (String): The File to be Processed
    """

    filename = OUTPUT_FILE_PATH + str(filename)
    arr = []
    # Using load txt() to write data from csv file to arr list
    arr = np.loadtxt(filename, delimiter=",", dtype=str)

    # X_list to store current time along with keypoints
    # Y_list to store the activity corresponding to the current time
    x_list = []
    y_list = []
    for data in arr:
        line = []
        flag = 0
        for i, value in enumerate(data):
            if value == '':
                flag = 1
                break
            elif i == MODEL_INPUT:
                y_list.append(value)
            else:
                line.append(float(value))
        if flag == 0:
            x_list.append(line)

    # converting list to Numpy array
    x = np.array(x_list)
    y = np.array(y_list)

    # Use LabelEncoder to convert string labels to integers
    label_encoder = LabelEncoder()
    label_encoder.fit(LABELS)

    # Used to encode the labels
    y = label_encoder.transform(y)
    y = to_categorical(y, num_classes=CLASS_OUTPUT)
    return x, y
