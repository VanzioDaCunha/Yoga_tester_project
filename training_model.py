import tensorflow as tf
import numpy as np
from model_input import data_preprocessing
from LSTM_Model import create_model
from keras.callbacks import EarlyStopping


sequence_length = 4
num_features = 133

train_files = []
train_set, train_labels = data_preprocessing('output1.csv')

for i in train_files:
    a, b = data_preprocessing(i)
    train_set = np.concatenate((train_set, a))
    train_labels = np.concatenate((train_labels, b))

test_files = []
test_set, test_labels = data_preprocessing('output2.csv')

for i in test_files:
    a, b = data_preprocessing(i)
    test_set = np.concatenate((test_set, a))
    test_labels = np.concatenate((test_labels, b))

num_samples = train_set.shape[0] // 4  # Calculate the number of samples after aggregation
train_set = train_set[:num_samples*4].reshape(-1, 4, 133)

num_samples = train_labels.shape[0] // 4  # Calculate the number of samples after aggregation
train_labels = train_labels[:num_samples*4].reshape(-1, 4, 9)


ip_shape = (sequence_length, num_features)
classifier = create_model(ip_shape)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

train_history = classifier.fit(x=train_set, y=train_labels, epochs=50, batch_size=4,
                               shuffle=False, validation_split=0.2,
                               callbacks=[early_stopping_callback])

num_samples = test_set.shape[0] // 4  # Calculate the number of samples after aggregation
test_set = test_set[:num_samples*4].reshape(-1, 4, 133)

num_samples = test_labels.shape[0] // 4  # Calculate the number of samples after aggregation
test_labels = test_labels[:num_samples*4].reshape(-1, 4, 9)

test_history = classifier.evaluate(test_set, test_labels)

classifier.save('modelname')
