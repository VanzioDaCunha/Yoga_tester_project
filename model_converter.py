import tensorflow as tf
from keras.models import load_model

# Load the Keras model
keras_model = load_model("Trikonasana2.keras")

# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False  # Disable experimental lowering of tensor list ops
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open("Trikonasana2.tflite", "wb") as f:
    f.write(tflite_model)
