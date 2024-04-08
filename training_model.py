import numpy as np
from model_input import data_preprocessing
from LSTM_Model import create_model
from keras.callbacks import EarlyStopping
from constants import MODEL_INPUT, CLASS_OUTPUT, SEQUENCE_LENGTH, LABELS, MODEL_LINK
from graph import plot_history, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf

# Used to remove some of the errors displayed by tensorflow
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Gets Model values from constants file
sequence_length = SEQUENCE_LENGTH
num_features = MODEL_INPUT

# Giving the file numbers to train dataset
train_set = []
train_labels = []

train_files = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 21,
               22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35]

# Extracting the set and labels from the dataset
for i in train_files:
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

train_set, test_set, train_labels, test_labels = train_test_split(train_set, train_labels, test_size=0.25, random_state=29)

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
#classifier.save(MODEL_LINK)
