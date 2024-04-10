from constants import MODEL_INPUT, CLASS_OUTPUT, SEQUENCE_LENGTH
from prepare_dataset import batched_dataset
from sklearn.model_selection import ParameterGrid
from keras.optimizers import Adam, RMSprop
from keras.losses import categorical_crossentropy as sparse
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Reshape
import tensorflow as tf
import heapq


def create_model(input_shape, optimizer='adam', filters=32, activation='relu', nodes=64):
    model = Sequential()

    # Define CNN-LSTM architecture
    model.add(Reshape((SEQUENCE_LENGTH, MODEL_INPUT, 1), input_shape=input_shape))
    model.add(TimeDistributed(Conv1D(filters=filters, kernel_size=3, activation=activation)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(units=nodes, return_sequences=True))
    model.add(Dense(units=CLASS_OUTPUT, activation='softmax'))

    # Compile the model
    model.compile(optimizer=optimizer, loss=sparse, metrics=['accuracy'])

    return model


# Used to remove some of the errors displayed by tensorflow
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Gets Model values from constants file
sequence_length = SEQUENCE_LENGTH
num_features = MODEL_INPUT

# retrieve the dataset from the csv files
train_set, test_set, train_labels, test_labels = batched_dataset()

# Define the parameter grid to search
param_grid = {
    'batch_size': [16, 32],
    'filters': [16, 32, 64],
    'nodes': [16, 32, 64, 128],
    'activation': ['relu', 'tanh'],
    'optimizer': ['adam', 'rmsprop']
}

best_results_heap = []

# Define a custom training loop for hyper parameter tuning
for params in ParameterGrid(param_grid):
    print("Testing parameters:", params)

    # Create model
    model = create_model((SEQUENCE_LENGTH, MODEL_INPUT), params['optimizer']
                         , params['filters'], params['activation'], params['nodes'])

    # Train model with current hyper parameters
    model.fit(train_set, train_labels, batch_size=params['batch_size'], epochs=50, verbose=0)

    # Evaluate model
    loss, accuracy = model.evaluate(test_set, test_labels, verbose=0)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

    # Add current result to heap
    heapq.heappush(best_results_heap, (accuracy, params))

    # Keep only the best 3 results
    if len(best_results_heap) > 3:
        heapq.heappop(best_results_heap)

# Print the best 3 results
print("Best 3 results:")
for i, (acc, params) in enumerate(sorted(best_results_heap, reverse=True)):
    print(f"Rank {i + 1}: Accuracy={acc}, Parameters={params}")