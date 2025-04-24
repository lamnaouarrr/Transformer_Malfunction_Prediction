# app/tf_compatibility.py

import tensorflow as tf

# Add missing functions for compatibility
if not hasattr(tf.keras.losses, 'mean_squared_error'):
    # Create a function that matches the API expected by baseline_fnn.py
    def mean_squared_error(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # Add the function to tf.keras.losses
    tf.keras.losses.mean_squared_error = mean_squared_error

# This allows the module to verify if patching was successful
def is_patched():
    return hasattr(tf.keras.losses, 'mean_squared_error')