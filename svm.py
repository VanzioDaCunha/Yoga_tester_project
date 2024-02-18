import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import iris  # For example dataset

# Load an example dataset or prepare your own exercise data
(X_train, y_train), (X_test, y_test) = iris.load_data()

# Preprocess input features (normalization, scaling, etc.)
# Adapt preprocessing based on your exercise data
X_train_scaled = tf.keras.utils.normalize(X_train, axis=-1)
X_test_scaled = tf.keras.utils.normalize(X_test, axis=-1)

# One-hot encode target labels if necessary
# Adapt encoding based on your exercise categories
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=10)  # Adjust num_classes
y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define a simple Sequential model for exercise classification
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # Adjust output units for your classes
])

# Compile the model with appropriate loss, optimizer, and metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model using early stopping for regularization
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train_scaled, y_train_encoded, epochs=10, validation_data=(X_test_scaled, y_test_encoded), callbacks=[early_stopping])

# Evaluate the trained model on the test set
test_loss, test_acc = model.evaluate(X_test_scaled, y_test_encoded)
print('Test accuracy:', test_acc)
