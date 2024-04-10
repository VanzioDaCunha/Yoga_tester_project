import numpy as np
from LSTM_Model import create_model
from keras.callbacks import EarlyStopping
from constants import MODEL_INPUT, CLASS_OUTPUT, SEQUENCE_LENGTH, LABELS, MODEL_LINK, OUTPUT_FILE_PATH
from graph import plot_history, plot_confusion_matrix
from prepare_dataset import batched_dataset
from sklearn import metrics
import tensorflow as tf

# Used to remove some of the errors displayed by tensorflow
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Gets Model values from constants file
sequence_length = SEQUENCE_LENGTH
num_features = MODEL_INPUT

# retrieve the dataset from the csv files
train_set, test_set, train_labels, test_labels = batched_dataset()

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
