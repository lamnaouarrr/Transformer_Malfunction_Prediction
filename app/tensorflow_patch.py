# tensorflow_patch.py
import tensorflow as tf

def apply_patch():
    """Apply patches to make newer TensorFlow versions compatible with older code"""
    # Add mean_squared_error function if it doesn't exist or isn't directly accessible
    if not hasattr(tf.keras.losses, 'mean_squared_error'):
        # Use the existing MeanSquaredError class to create a compatible function
        def mean_squared_error(y_true, y_pred):
            mse = tf.keras.losses.MeanSquaredError()
            return mse(y_true, y_pred)
        
        # Add the function to the module
        tf.keras.losses.mean_squared_error = mean_squared_error
        print("✓ Applied patch for tf.keras.losses.mean_squared_error")
    else:
        print("✓ mean_squared_error already exists in tf.keras.losses")
    
    return True