import tensorflow as tf

print("TF:", tf.__version__)
print("All devices:", tf.config.list_physical_devices())
print("GPU devices:", tf.config.list_physical_devices("GPU"))