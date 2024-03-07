import numpy as np
from model_input import data_preprocessing
from LSTM_Model import create_model
from keras.callbacks import EarlyStopping
from constants import MODEL_INPUT, CLASS_OUTPUT, SEQUENCE_LENGTH
from graph import plot_history

sequence_length = SEQUENCE_LENGTH
num_features = MODEL_INPUT

train_files = ['output3.csv', 'output4.csv', 'output5.csv', 'output7.csv', 'output12.csv',
               'output13.csv', 'output2.csv', 'output8.csv', 'output10.csv']
train_set, train_labels = data_preprocessing('output1.csv')

for i in train_files:
    a, b = data_preprocessing(i)
    train_set = np.concatenate((train_set, a))
    train_labels = np.concatenate((train_labels, b))

test_files = []
test_set, test_labels = data_preprocessing('output11.csv')

for i in test_files:
    a, b = data_preprocessing(i)
    test_set = np.concatenate((test_set, a))
    test_labels = np.concatenate((test_labels, b))

num_samples = train_set.shape[0] // SEQUENCE_LENGTH  # Calculate the number of samples after aggregation
train_set = train_set[:num_samples * SEQUENCE_LENGTH].reshape(-1, SEQUENCE_LENGTH, MODEL_INPUT)

num_samples = train_labels.shape[0] // SEQUENCE_LENGTH  # Calculate the number of samples after aggregation
train_labels = train_labels[:num_samples * SEQUENCE_LENGTH].reshape(-1, SEQUENCE_LENGTH, CLASS_OUTPUT)

num_samples = train_set.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)

# Use the shuffled indices to shuffle the data
train_set = train_set[indices]
train_labels = train_labels[indices]

ip_shape = (sequence_length, num_features)
classifier = create_model(ip_shape)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=30, mode='min', restore_best_weights=False)

train_history = classifier.fit(x=train_set, y=train_labels, epochs=150, batch_size=4,
                               shuffle=False, validation_split=0.2,
                               callbacks=[early_stopping_callback])
plot_history(train_history)

num_samples = test_set.shape[0] // SEQUENCE_LENGTH  # Calculate the number of samples after aggregation
test_set = test_set[:num_samples * SEQUENCE_LENGTH].reshape(-1, SEQUENCE_LENGTH, MODEL_INPUT)

num_samples = test_labels.shape[0] // SEQUENCE_LENGTH  # Calculate the number of samples after aggregation
test_labels = test_labels[:num_samples * SEQUENCE_LENGTH].reshape(-1, SEQUENCE_LENGTH, CLASS_OUTPUT)

test_history = classifier.evaluate(test_set, test_labels)

classifier.save('Trikonasana.keras')
