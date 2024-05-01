import tensorflow as tf
from keras.models import load_model
from constants import MODEL_LINK, TF_MODEL_LINK

# Load the Keras model
keras_model = load_model(MODEL_LINK)

# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False  # Disable experimental lowering of tensor list ops
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open(TF_MODEL_LINK, "wb") as f:
    f.write(tflite_model)
