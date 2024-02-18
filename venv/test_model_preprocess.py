import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def data_preprocessing(filename):
    arr = np.loadtxt(filename, delimiter=",", dtype=str)  # Read data as strings

    X_list = []
    Y_list = []
    for data in arr:
        line = []
        flag = 0
        for i, value in enumerate(data):
            if value == '':  # Skip empty values
                flag = 1
                break
            elif i == 133:  # Last column, assumed to contain labels
                Y_list.append(value)
            else:
                try:
                    line.append(float(value))  # Convert numeric values to float
                except ValueError:
                    flag = 1  # Handle the case where conversion is not possible
                    break
        if flag == 0:
            X_list.append(line)

    x = np.array(X_list, dtype=float)  # Ensure the array is of type float
    y = np.array(Y_list)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # Convert labels to integers
    y = to_categorical(y)  # Then apply one-hot encoding

    return x, y
