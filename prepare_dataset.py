import numpy as np
import os
from model_input import data_preprocessing
from constants import MODEL_INPUT, SEQUENCE_LENGTH, OUTPUT_FILE_PATH
from sklearn.model_selection import train_test_split


def batched_dataset():
    """
            This returns the dataset from hte csv files in batched format for the cnn-lstm model
    """
    # Gets Model values from constants file
    sequence_length = SEQUENCE_LENGTH

    # Giving the file numbers to train dataset
    train_set = []
    train_labels = []

    folder_path = OUTPUT_FILE_PATH
    folder = [filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]

    # Extracting the set and labels from the dataset
    for i in folder:
        a, b = data_preprocessing(i)

        remove_elements = len(a) % sequence_length
        if remove_elements > 0:
            a = a[:-remove_elements]
            b = b[:-remove_elements]

        num_batch = len(a) // SEQUENCE_LENGTH
        data_batch = np.array_split(a, num_batch)
        train_set.extend(data_batch)
        data_labels = np.array_split(b, num_batch)
        train_labels.extend(data_labels)

    train_set = np.array(train_set)
    train_labels = np.array(train_labels)

    train_set, test_set, train_labels, test_labels = train_test_split(train_set, train_labels,
                                                                      test_size=0.25, random_state=29)
    return train_set, test_set, train_labels, test_labels
