import numpy as np
from model_input import data_preprocessing
from constants import MODEL_INPUT, SEQUENCE_LENGTH, LABELS
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from graph import plot_confusion_matrix

# Evaluate accuracy

# Used to remove some of the errors displayed by tensorflow
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Gets Model values from constants file
sequence_length = SEQUENCE_LENGTH
num_features = MODEL_INPUT

# Giving the file numbers to train dataset
all_files = [2, 3, 4, 5, 6, 7, 8, 10, 12, 21,
             24, 25, 26, 27, 30, 32]
train_set, train_labels = data_preprocessing(1)

# Extracting the set and labels from the dataset
for i in all_files:
    a, b = data_preprocessing(i)
    train_set = np.concatenate((train_set, a))
    train_labels = np.concatenate((train_labels, b))

print(train_set.shape)
print(train_labels.shape)

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
