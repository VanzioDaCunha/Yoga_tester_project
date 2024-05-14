import tensorflow as tf
from keras.models import load_model
from constants import MODEL_LINK, TF_MODEL_LINK

# Load the Keras model
keras_model = load_model(MODEL_LINK)

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open(TF_MODEL_LINK, "wb").write(tflite_model)