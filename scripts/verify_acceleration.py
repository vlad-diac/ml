import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # Should show Metal device(s)
print(tf.__version__)