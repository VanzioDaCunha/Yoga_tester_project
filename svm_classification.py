import numpy as np
import os
from model_input import data_preprocessing
from constants import MODEL_INPUT, SEQUENCE_LENGTH, LABELS, OUTPUT_FILE_PATH
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from graph import plot_confusion_matrix


# Used to remove some of the errors displayed by tensorflow
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Gets Model values from constants file
sequence_length = SEQUENCE_LENGTH
num_features = MODEL_INPUT

# Giving the files to the dataset
train_set = []
train_labels = []
folder_path = OUTPUT_FILE_PATH
folder = [filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]

# Extracting the set and labels from the dataset
for i in folder:
    a, b = data_preprocessing(i)
    train_set.extend(a)
    train_labels.extend(b)

train_set = np.array(train_set)
train_labels = np.array(train_labels)

if len(train_labels.shape) > 1:
    train_labels = np.argmax(train_labels, axis=1)

X_train, X_test, y_train, y_test = train_test_split(train_set, train_labels, test_size=0.25, random_state=29)

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Evaluate precision
precision = precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)

# Evaluate recall
recall = recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)

# Evaluate F1-score
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1-score:", f1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")

plot_confusion_matrix(conf_matrix, LABELS, normalize=True)
