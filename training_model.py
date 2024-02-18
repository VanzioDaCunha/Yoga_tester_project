import tensorflow as tf
import numpy as np
from model_input import data_preprocessing
from LSTM_Model import create_model
from keras.callbacks import EarlyStopping
from keras.utils import plot_model


sequence_length = 4
num_features = 133

train_files = ['output.csv']
train_set, train_labels = data_preprocessing('output.csv')

for i in train_files:
    a, b = data_preprocessing(i)
    train_set = np.concatenate((train_set, a))
    train_labels = np.concatenate((train_labels, b))

test_files = ['output.csv', 'output.csv']
test_set, test_labels = data_preprocessing('output.csv')

for i in test_files:
    a, b = data_preprocessing(i)
    test_set = np.concatenate((test_set, a))
    test_labels = np.concatenate((test_labels, b))

num_samples = train_set.shape[0] // 4  # Calculate the number of samples after aggregation
train_set = train_set[:num_samples*4].reshape(-1, 4, 133)


print(train_labels.shape)

num_samples = train_labels.shape[0] // 4  # Calculate the number of samples after aggregation
train_labels = train_labels[:num_samples*4].reshape(-1, 4, 8)

print(train_labels.shape)
print(train_labels)

ip_shape = (sequence_length, num_features)
classifier = create_model(ip_shape)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

train_history = classifier.fit(x=train_set, y=train_labels, epochs=50, batch_size=4,
                               shuffle=False, validation_split=0.2,
                               callbacks=[early_stopping_callback])
