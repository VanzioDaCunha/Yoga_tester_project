import numpy as np
from model_input import data_preprocessing
from LSTM_Model import create_model
from keras.callbacks import EarlyStopping
from constants import MODEL_INPUT, CLASS_OUTPUT, SEQUENCE_LENGTH, LABELS, MODEL_LINK
from graph import plot_history, plot_confusion_matrix
from sklearn import metrics
import tensorflow as tf

# Used to remove some of the errors displayed by tensorflow
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Gets Model values from constants file
sequence_length = SEQUENCE_LENGTH
num_features = MODEL_INPUT

# Giving the file numbers to train dataset
train_files = [2, 3, 4, 6, 7, 8, 10, 11, 13, 5, 16]
train_set, train_labels = data_preprocessing(1)

# Extracting the set and labels from the dataset
for i in train_files:
    a, b = data_preprocessing(i)
    train_set = np.concatenate((train_set, a))
    train_labels = np.concatenate((train_labels, b))

# Giving the file numbers to test dataset
test_files = [15, 14]
test_set, test_labels = data_preprocessing(12)

# Extracting the set and labels from the dataset
for i in test_files:
    a, b = data_preprocessing(i)
    test_set = np.concatenate((test_set, a))
    test_labels = np.concatenate((test_labels, b))

# Reshaping the data to be feed into the model
num_samples = train_set.shape[0] // SEQUENCE_LENGTH
train_set = train_set[:num_samples * SEQUENCE_LENGTH].reshape(-1, SEQUENCE_LENGTH, MODEL_INPUT)
num_samples = train_labels.shape[0] // SEQUENCE_LENGTH
train_labels = train_labels[:num_samples * SEQUENCE_LENGTH].reshape(-1, SEQUENCE_LENGTH, CLASS_OUTPUT)

# Shuffling the train dataset
num_samples = train_set.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)

# Use the shuffled indices to shuffle the data
train_set = train_set[indices]
train_labels = train_labels[indices]

# Creating the Model from saved file
ip_shape = (sequence_length, num_features)
classifier = create_model(ip_shape)

# setting an early callback to stop to avoid over fitting
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=25, mode='min', restore_best_weights=True)

# Training the model to train data
train_history = classifier.fit(x=train_set, y=train_labels, epochs=150, batch_size=16,
                               shuffle=True, validation_split=0.20,
                               callbacks=[early_stopping_callback])

# plotting the graph data for training
plot_history(train_history)

# Reshaping the test data to be feed into the model
num_samples = test_set.shape[0] // SEQUENCE_LENGTH
test_set = test_set[:num_samples * SEQUENCE_LENGTH].reshape(-1, SEQUENCE_LENGTH, MODEL_INPUT)
num_samples = test_labels.shape[0] // SEQUENCE_LENGTH
test_labels = test_labels[:num_samples * SEQUENCE_LENGTH].reshape(-1, SEQUENCE_LENGTH, CLASS_OUTPUT)

# Shuffling the test dataset
num_samples = test_set.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)

# Use the shuffled indices to shuffle the data
test_set = test_set[indices]
test_labels = test_labels[indices]

# testing unknown data on the model
test_history = classifier.evaluate(test_set, test_labels)

# getting the predicted labels for the test set
y_pred = classifier.predict(test_set)
y_pred = np.reshape(y_pred, (-1, CLASS_OUTPUT))
y_pred = np.argmax(y_pred, axis=1)

# reshaping the true labels to find confusion matrix
y_true = test_labels
y_true = np.reshape(y_true, (-1, CLASS_OUTPUT))
y_true = np.argmax(y_true, axis=1)

# printing confusion matrix and report
cm = metrics.confusion_matrix(y_true, y_pred)
print(metrics.classification_report(y_true, y_pred, digits=3))

# confusion matrix
plot_confusion_matrix(cm, LABELS, normalize=True)

# Saving the model in keras format
classifier.save(MODEL_LINK)
