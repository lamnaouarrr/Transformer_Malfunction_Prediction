#!/usr/bin/env python3
"""
simple_ultra_ast_trainer.py - Simplified version of the AST trainer for RunPod environment

This version has minimal dependencies and simplified data loading for a RunPod VPS
"""

import os
import sys
import time
import pickle
import yaml
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import AdamW
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

# Create required directories
os.makedirs("./model/AST", exist_ok=True)
os.makedirs("./result/result_AST", exist_ok=True)
os.makedirs("./logs/log_AST", exist_ok=True)

# Configure TensorFlow for memory efficiency
print("Configuring TensorFlow...")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        try:
            # Limit memory growth
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Set memory growth for {device}")
        except Exception as e:
            print(f"Error setting memory growth: {e}")
else:
    print("No GPU found. Using CPU for training.")

# Enable XLA compilation for faster execution
tf.config.optimizer.set_jit(True)
print("XLA compilation enabled")

# Load configuration file
try:
    with open("baseline_AST.yaml", "r") as file:
        param = yaml.safe_load(file)
        print("Configuration loaded successfully")
except Exception as e:
    print(f"Error loading configuration: {e}")
    sys.exit(1)

# Enable mixed precision if specified in config
if param.get("training", {}).get("mixed_precision", True):
    mixed_precision.set_global_policy('mixed_float16')
    print(f"Mixed precision policy set to: {mixed_precision.global_policy()}")

# Clean memory function
def clean_memory():
    """Aggressively clean memory"""
    gc.collect()
    tf.keras.backend.clear_session()
    print("Memory cleaned")

# Define smaller target spectrogram shape to reduce memory usage
target_freq_dim = param["feature"]["n_mels"]  # From config
target_time_dim = 96  # Fixed value that balances information vs memory usage
target_shape = (target_freq_dim, target_time_dim)
print(f"Target spectrogram shape: {target_shape}")

# Preprocess spectrogram function
def preprocess_spectrogram(spec, target_shape):
    """Process a spectrogram to have consistent dimensions"""
    # Handle 3D input if needed
    if len(spec.shape) == 3 and spec.shape[2] == 1:
        spec = spec[:, :, 0]
    
    # Skip if dimensions already match
    if spec.shape[0] == target_shape[0] and spec.shape[1] == target_shape[1]:
        return spec
    
    # For memory efficiency, use simple padding/cropping    
    result = np.zeros(target_shape, dtype=np.float16)  # Use float16 to reduce memory usage
    
    # Copy data with bounds checking
    freq_dim = min(spec.shape[0], target_shape[0])
    time_dim = min(spec.shape[1], target_shape[1])
    result[:freq_dim, :time_dim] = spec[:freq_dim, :time_dim]
    
    return result

# Create a simple AST model - adjusted for RunPod environment
def create_simple_ast_model(input_shape):
    """Create a simplified AST model that will work on lower memory environments"""
    print(f"Creating simplified AST model with input shape: {input_shape}")
    
    # Number of transformer layers
    num_layers = 1
    
    # Define input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Add channel dimension
    x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    # Apply 2D convolution to reduce dimensions
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=4,
        strides=4,
        padding='valid',
        activation='relu',
        name='patch_embedding'
    )(x)
    
    # Reshape to sequence format
    x = tf.keras.layers.Reshape((-1, 32))(x)
    
    # Apply layer normalization
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Add simple self-attention layer
    for i in range(num_layers):
        # Self-attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=1,
            key_dim=32,
            dropout=0.1
        )(x, x)
        
        # Add residual connection
        x = tf.keras.layers.Add()([x, attention])
        
        # Layer normalization
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn = tf.keras.layers.Dense(32, activation='gelu')(x)
        ffn = tf.keras.layers.Dense(32)(ffn)
        
        # Add residual connection
        x = tf.keras.layers.Add()([x, ffn])
        
        # Layer normalization
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = tf.keras.layers.Dense(16, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create and return model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    print(f"Model created with {model.count_params()} parameters")
    
    return model

# Calculate dataset statistics (mean and std)
def get_dataset_statistics(train_pickle, target_shape, sample_size=1000):
    """Calculate or load dataset statistics for normalization"""
    print("Calculating dataset statistics...")
    
    try:
        # Load a sample of training data
        with open(train_pickle, 'rb') as f:
            data = pickle.load(f)
            # Limit to sample_size
            sample_size = min(sample_size, len(data))
            data_sample = data[:sample_size]
        
        # Process sample
        processed_sample = np.zeros((sample_size, target_shape[0], target_shape[1]), dtype=np.float16)
        for i in range(sample_size):
            processed_sample[i] = preprocess_spectrogram(data_sample[i], target_shape)
        
        # Calculate statistics
        mean = np.mean(processed_sample)
        std = np.std(processed_sample)
        
        # Clean memory
        del data_sample
        del processed_sample
        clean_memory()
        
        print(f"Dataset statistics: mean={mean:.4f}, std={std:.4f}")
        
        # Handle infinity or zero values
        if not np.isfinite(std) or std == 0:
            print("Warning: Standard deviation is infinite or zero. Using 1.0 instead.")
            std = 1.0
            
        return mean, std
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        # Return safe defaults
        return 0.0, 1.0

# Main function to run training
def main():
    print("Starting simplified ultra memory-efficient AST training")
    
    # Define paths to data
    pickle_dir = param['pickle_directory']
    train_pickle = f"{pickle_dir}/train_overall.pickle"
    train_labels_pickle = f"{pickle_dir}/train_labels_overall.pickle"
    val_pickle = f"{pickle_dir}/val_overall.pickle"
    val_labels_pickle = f"{pickle_dir}/val_labels_overall.pickle"
    test_pickle = f"{pickle_dir}/test_files_overall.pickle"
    test_labels_pickle = f"{pickle_dir}/test_labels_overall.pickle"
    
    print(f"Using data from: {pickle_dir}")
    
    # Check if pickle files exist
    for p in [train_pickle, train_labels_pickle, val_pickle, val_labels_pickle, test_pickle, test_labels_pickle]:
        if not os.path.exists(p):
            print(f"Error: Missing pickle file {p}")
            sys.exit(1)
    
    # Get dataset statistics for normalization
    mean, std = get_dataset_statistics(train_pickle, target_shape)
    
    # Create model
    print("Creating model...")
    model = create_simple_ast_model(input_shape=target_shape)
    
    # Print model summary
    model.summary()
    
    # Configure model for training
    print("Configuring model for training...")
    optimizer = AdamW(
        learning_rate=param.get("fit", {}).get("compile", {}).get("learning_rate", 0.0001),
        weight_decay=0.01,
        clipnorm=param.get("training", {}).get("gradient_clip_norm", 1.0)
    )
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC()
        ]
    )
    
    # Setup callback list
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        # Learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-8
        ),
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{param['model_directory']}/best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        ),
        # Add tensorboard callback for monitoring
        tf.keras.callbacks.TensorBoard(
            log_dir=f"./logs/log_AST/tensorboard",
            histogram_freq=1
        )
    ]
    
    # Create data generators directly with tf.data.Dataset
    print("Creating data generators...")
    
    # Function to load and process data
    def process_data(data_pickle, labels_pickle, batch_size, is_training=True):
        print(f"Loading data from {data_pickle} and {labels_pickle}")
        
        # Load data and labels
        with open(data_pickle, 'rb') as f:
            data = pickle.load(f)
        
        with open(labels_pickle, 'rb') as f:
            labels = pickle.load(f)
        
        # Convert to arrays for efficiency
        process_for_batch = lambda x, y: (preprocess_spectrogram(x, target_shape), y)
        
        # Create datasets
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        
        # Apply preprocessing
        dataset = dataset.map(
            lambda x, y: tf.py_function(
                process_for_batch, [x, y], [tf.float16, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Reshape appropriately
        dataset = dataset.map(
            lambda x, y: (tf.reshape(x, target_shape), y)
        )
        
        # Apply normalization
        dataset = dataset.map(
            lambda x, y: ((x - mean) / std, y)
        )
        
        # Shuffle if training
        if is_training:
            dataset = dataset.shuffle(buffer_size=10000)
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    # Create training and validation datasets
    try:
        batch_size = param.get("fit", {}).get("batch_size", 8)
        # Use smaller batch size to avoid OOM
        actual_batch_size = max(1, batch_size // 2)
        
        print(f"Using batch size: {actual_batch_size} (reduced from {batch_size} to avoid OOM)")
        
        # Create datasets
        train_dataset = process_data(train_pickle, train_labels_pickle, actual_batch_size, is_training=True)
        val_dataset = process_data(val_pickle, val_labels_pickle, actual_batch_size, is_training=False)
        
        # Start training
        print("Starting training...")
        start_time = time.time()
        
        history = model.fit(
            train_dataset,
            epochs=param.get("fit", {}).get("epochs", 50),
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # End timing
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Save final model
        model.save(f"{param['model_directory']}/final_model.keras")
        print("Final model saved")
        
        # Plot training history
        plt.figure(figsize=(12, 8))
        
        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(history.history['binary_accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{param['result_directory']}/training_history.png")
        plt.close()
        
        print("Training history plot saved")
        
        # Evaluate on test data
        print("\nEvaluating model on test set")
        test_dataset = process_data(test_pickle, test_labels_pickle, actual_batch_size, is_training=False)
        
        # Create predictions list
        y_true = []
        y_pred = []
        
        # Prediction loop
        for x_batch, y_batch in test_dataset:
            # Get predictions
            pred_batch = model.predict(x_batch)
            
            # Store true labels and predictions
            y_true.extend(y_batch.numpy())
            y_pred.extend(pred_batch.flatten())
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate classification metrics
        y_pred_binary = (y_pred > 0.5).astype(int)
        classification_rep = classification_report(y_true, y_pred_binary)
        conf_matrix = confusion_matrix(y_true, y_pred_binary)
        auc = roc_auc_score(y_true, y_pred)
        
        # Print evaluation results
        print("\nTest Evaluation Results:")
        print(f"AUC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_rep)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        # Save results to text file
        with open(f"{param['result_directory']}/test_evaluation.txt", 'w') as f:
            f.write("Test Evaluation Results:\n")
            f.write(f"AUC: {auc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_rep)
            f.write("\nConfusion Matrix:\n")
            f.write(str(conf_matrix))
        
        print("Test evaluation results saved")
        print("\nTraining and evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Run the main function
if __name__ == "__main__":
    main()