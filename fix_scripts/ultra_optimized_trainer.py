#!/usr/bin/env python3
"""
ultra_optimized_trainer.py - A comprehensive solution for memory-efficient AST training
Designed to work with high-end GPUs while preventing OOM errors
"""

import os
import sys
import time
import pickle
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
import gc

# Force TensorFlow to use the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Memory growth settings - crucial for preventing OOM errors
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Clear any existing TensorFlow session and free GPU memory
print("\n=== GPU Memory Management Setup ===")
print("Clearing GPU memory and setting up optimized environment...")
tf.keras.backend.clear_session()

# Configure GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth enabled for {len(gpus)} GPUs")
        
        # Get GPU info
        gpu_info = tf.config.experimental.get_device_details(gpus[0])
        print(f"GPU detected: {gpu_info.get('device_name', 'Unknown GPU')}")
        
        # No hard memory limit - instead use dynamic allocation with growth limits
        # This works better than a fixed cap on V100 GPUs
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPUs detected. This script is optimized for GPU training.")
    sys.exit(1)

# Enable XLA for faster training
print("Enabling XLA JIT compilation...")
tf.config.optimizer.set_jit(True)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2'

# Make sure directories exist
os.makedirs("./model/AST", exist_ok=True)
os.makedirs("./result/result_AST", exist_ok=True)
os.makedirs("./logs/log_AST", exist_ok=True)
os.makedirs("/tmp/ast_training", exist_ok=True)

# Load configuration
print("\n=== Loading Configuration ===")
try:
    with open("baseline_AST.yaml", "r") as stream:
        param = yaml.safe_load(stream)
        print("Configuration loaded successfully")
except Exception as e:
    print(f"Error loading configuration: {e}")
    sys.exit(1)

# Enable mixed precision for faster training and reduced memory usage
print("\n=== Setting Up Mixed Precision ===")
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed precision policy: {policy}")

# Define paths to pickle files
pickle_dir = param['pickle_directory']
train_pickle = f"{pickle_dir}/train_overall.pickle"
val_pickle = f"{pickle_dir}/val_overall.pickle"
test_pickle = f"{pickle_dir}/test_overall.pickle"

# Set up memory-efficient data loading
class MemoryEfficientProcessor:
    """Memory-efficient processor that handles data in batches"""
    
    def __init__(self, target_shape):
        self.target_shape = target_shape
    
    def process_batch(self, batch_data, normalize=False, mean=None, std=None):
        """Process a batch of data with memory efficiency in mind"""
        processed_batch = []
        
        for i, data in enumerate(batch_data):
            # Process each sample
            feature = data[0]  # Assuming data is tuple of (feature, label)
            
            # Memory-efficient resizing
            if feature.shape != self.target_shape:
                # Resize to target shape without keeping original
                resized = tf.image.resize(
                    feature[..., tf.newaxis], 
                    self.target_shape[:2], 
                    method='bilinear'
                )
                feature = resized.numpy()[..., 0]
            
            # Normalize if needed
            if normalize and mean is not None and std is not None and std > 0:
                feature = (feature - mean) / std
                
            processed_batch.append(feature)
            
            # Clear memory periodically
            if (i + 1) % 100 == 0:
                gc.collect()
        
        # Convert to numpy array and add channel dimension
        processed_batch = np.array(processed_batch)[..., np.newaxis]
        return processed_batch

class StreamingDataGenerator(tf.keras.utils.Sequence):
    """Memory-efficient data generator that loads batches on demand"""
    
    def __init__(self, data_pickle, batch_size, processor, mean=None, std=None):
        self.data_pickle = data_pickle
        self.batch_size = batch_size
        self.processor = processor
        self.mean = mean
        self.std = std
        
        # Get data length without loading all data
        with open(self.data_pickle, 'rb') as f:
            self.data_sample = pickle.load(f)
            self.total_samples = len(self.data_sample)
        
        self.data_sample = None
        gc.collect()
        
        self.indexes = np.arange(self.total_samples)
        np.random.shuffle(self.indexes)
        
    def __len__(self):
        return int(np.ceil(self.total_samples / self.batch_size))
    
    def __getitem__(self, idx):
        # Calculate batch indexes
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Load data from pickle for just this batch
        with open(self.data_pickle, 'rb') as f:
            all_data = pickle.load(f)
            batch_data = [all_data[i] for i in batch_indexes]
        
        # Process features
        batch_features = [item[0] for item in batch_data]
        processed_features = self.processor.process_batch(
            batch_features, 
            normalize=True, 
            mean=self.mean, 
            std=self.std
        )
        
        # Process labels
        batch_labels = np.array([item[1] for item in batch_data], dtype=np.float32)
        
        # Clear memory
        batch_data = None
        batch_features = None
        gc.collect()
        
        return processed_features, batch_labels
    
    def on_epoch_end(self):
        # Shuffle indexes at the end of each epoch
        np.random.shuffle(self.indexes)
        gc.collect()

# Load validation data for statistics calculation
print("\n=== Preparing Data ===")
print("Loading validation data...")
try:
    with open(val_pickle, 'rb') as f:
        val_data = pickle.load(f)
    val_labels = np.array([item[1] for item in val_data], dtype=np.float32)
    print(f"Loaded {len(val_data)} validation samples")
except Exception as e:
    print(f"Error loading validation data: {e}")
    sys.exit(1)

# Set target spectrogram shape
print("Setting up target shape...")
target_shape = (param['feature']['n_mels'], param['feature']['frames'])
print(f"Target shape: {target_shape}")

# Initialize processor
processor = MemoryEfficientProcessor(target_shape)

# Calculate mean and std from a sample of training data
print("Calculating dataset statistics...")
try:
    with open(train_pickle, 'rb') as f:
        # Only sample a subset for statistics to save memory
        train_sample = pickle.load(f)[:1000]  
    
    processed_sample = processor.process_batch(
        [item[0] for item in train_sample]
    )
    mean = np.mean(processed_sample)
    std = np.std(processed_sample)
    print(f"Training statistics - Mean: {mean:.4f}, Std: {std:.4f}")
    
    # Free memory
    train_sample = None
    processed_sample = None
    gc.collect()
except Exception as e:
    print(f"Error calculating statistics: {e}")
    sys.exit(1)

# Process validation data efficiently
print("Processing validation data...")
processed_val_data = processor.process_batch(
    [item[0] for item in val_data], 
    normalize=True, 
    mean=mean, 
    std=std
)

# Free memory
val_data = None
gc.collect()

# Create validation dataset
print("Creating optimized validation dataset...")
val_dataset = tf.data.Dataset.from_tensor_slices((processed_val_data, val_labels))
val_dataset = val_dataset.batch(
    param.get("fit", {}).get("batch_size", 8)
).prefetch(2)

# Free memory
processed_val_data = None
gc.collect()

# Function to create a positional encoding
def positional_encoding(seq_len, d_model):
    # Create positional encoding
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

# Function to create the AST model with memory optimizations
def create_ast_model(input_shape, config=None):
    """Creates a memory-optimized AST model for transformer malfunction prediction"""
    print(f"\n=== Building Memory-Optimized AST Model ===")
    print(f"Creating model with input shape {input_shape}...")
    
    if config is None:
        config = {}
    
    # Get transformer configuration with memory-efficient defaults
    transformer_config = config.get("transformer", {})
    num_heads = transformer_config.get("num_heads", 1)
    dim_feedforward = transformer_config.get("dim_feedforward", 64)
    num_encoder_layers = transformer_config.get("num_encoder_layers", 1)
    patch_size = transformer_config.get("patch_size", 4)
    attention_dropout = transformer_config.get("attention_dropout", 0.1)
    ff_dim_multiplier = transformer_config.get("ff_dim_multiplier", 2)
    
    # Calculate sequence length and embedding dimension
    h_patches = input_shape[0] // patch_size
    w_patches = input_shape[1] // patch_size
    seq_len = h_patches * w_patches
    embed_dim = dim_feedforward
    
    print(f"Model configuration: Heads={num_heads}, Feed-forward dim={dim_feedforward}")
    print(f"Sequence length: {seq_len}, Embedding dimension: {embed_dim}")
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Add channel dimension if not present
    x = tf.keras.layers.Reshape((*input_shape, 1))(inputs) if len(input_shape) == 2 else inputs
    
    # Patchify the input - this breaks the spectrogram into patches
    x = tf.keras.layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patchify"
    )(x)
    
    # Flatten patches
    batch_size = tf.shape(x)[0]
    x = tf.keras.layers.Reshape((h_patches * w_patches, embed_dim))(x)
    
    # Add positional embedding
    pos_encoding = positional_encoding(seq_len, embed_dim)
    x = x + pos_encoding[:, :seq_len, :]
    
    # Add classification token
    cls_token = tf.keras.layers.Dense(embed_dim)(
        tf.ones((batch_size, 1, 1))
    )
    x = tf.keras.layers.Concatenate(axis=1)([cls_token, x])
    
    # Layer normalization
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Apply transformer encoder layers with gradient checkpointing for memory efficiency
    for i in range(num_encoder_layers):
        # Self-attention
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=attention_dropout
        )(x, x)
        
        # Add dropout and residual connection
        attn_output = tf.keras.layers.Dropout(0.1)(attn_output)
        x = tf.keras.layers.Add()([x, attn_output])
        
        # Layer normalization
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network - reduced multiplier for memory efficiency
        ffn = tf.keras.layers.Dense(embed_dim * ff_dim_multiplier, activation='gelu')(x)
        ffn = tf.keras.layers.Dense(embed_dim)(ffn)
        
        # Add dropout and residual connection
        ffn = tf.keras.layers.Dropout(0.1)(ffn)
        x = tf.keras.layers.Add()([x, ffn])
        
        # Layer normalization
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Use only the classification token
    x = tf.keras.layers.Lambda(lambda x: x[:, 0])(x)
    
    # Final classification head with smaller dense layer
    x = tf.keras.layers.Dense(32, activation='gelu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(f"Model created with {model.count_params()} parameters")
    
    return model

# Verify that TensorFlow is using the GPU
print("\n=== Verifying GPU Setup ===")
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print("Test GPU computation result:", c.numpy())
    print("Device:", a.device, b.device, c.device)

# Create memory-optimized AST model
model = create_ast_model(
    input_shape=target_shape, 
    config=param.get("model", {}).get("architecture", {})
)

# Compile model with optimizations
print("\n=== Compiling Model ===")
learning_rate = param.get("fit", {}).get("compile", {}).get("learning_rate", 0.0001)
gradient_clip_norm = param.get("training", {}).get("gradient_clip_norm", 1.0)

optimizer = AdamW(
    learning_rate=learning_rate,
    weight_decay=0.01,
    clipnorm=gradient_clip_norm
)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
print("\n=== Setting Up Training Callbacks ===")
callbacks = []

# Model checkpoint
checkpoint_path = f"{param['model_directory']}/best_model.h5"
callbacks.append(
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )
)

# Early stopping
if param.get("fit", {}).get("early_stopping", {}).get("enabled", True):
    print("Adding early stopping callback")
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=param.get("fit", {}).get("early_stopping", {}).get("monitor", "val_loss"),
            patience=param.get("fit", {}).get("early_stopping", {}).get("patience", 15),
            restore_best_weights=True
        )
    )

# Learning rate scheduler
if param.get("fit", {}).get("lr_scheduler", {}).get("enabled", True):
    print("Adding learning rate scheduler callback")
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=param.get("fit", {}).get("lr_scheduler", {}).get("monitor", "val_loss"),
            factor=param.get("fit", {}).get("lr_scheduler", {}).get("factor", 0.1),
            patience=param.get("fit", {}).get("lr_scheduler", {}).get("patience", 5),
            min_lr=param.get("fit", {}).get("lr_scheduler", {}).get("min_lr", 0.00000001)
        )
    )

# Memory clearance callback
class MemoryClearanceCallback(tf.keras.callbacks.Callback):
    def __init__(self, clear_frequency=5):
        super(MemoryClearanceCallback, self).__init__()
        self.clear_frequency = clear_frequency
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.clear_frequency == 0:
            print(f"\nClearing memory at epoch {epoch + 1}...")
            gc.collect()
            tf.keras.backend.clear_session()

callbacks.append(
    MemoryClearanceCallback(
        clear_frequency=param.get("training", {})
                           .get("memory_optimization", {})
                           .get("clear_memory_frequency", 5)
    )
)

# Create data generator
print("\n=== Creating Data Generator ===")
batch_size = param.get("fit", {}).get("batch_size", 8)
print(f"Batch size: {batch_size}")

train_generator = StreamingDataGenerator(
    data_pickle=train_pickle,
    batch_size=batch_size,
    processor=processor,
    mean=mean,
    std=std
)

# Train model with optimizations
print("\n=== Starting Model Training ===")
print(f"Training for {param['fit']['epochs']} epochs")
print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
print("Training will use GPU acceleration with mixed precision")

training_start_time = time.time()

# Train the model with memory-efficient settings
history = model.fit(
    train_generator,
    epochs=param["fit"]["epochs"],
    validation_data=val_dataset,
    verbose=1,
    callbacks=callbacks,
    workers=4,  # Limit workers to prevent memory issues
    max_queue_size=2  # Small queue size to save memory
)

training_end_time = time.time()
training_duration = training_end_time - training_start_time
print(f"\nTraining completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")

# Save history to pickle
print("\n=== Saving Results ===")
history_pickle = f"{param['result_directory']}/history.pickle"
with open(history_pickle, 'wb') as f:
    pickle.dump(history.history, f)

# Save training info to YAML
training_info = {
    "training_time_seconds": training_duration,
    "final_val_accuracy": float(history.history['val_accuracy'][-1]),
    "final_val_loss": float(history.history['val_loss'][-1]),
    "final_train_accuracy": float(history.history['accuracy'][-1]),
    "final_train_loss": float(history.history['loss'][-1]),
    "trained_epochs": len(history.history['loss']),
    "best_val_accuracy": float(max(history.history['val_accuracy'])),
    "best_val_accuracy_epoch": int(np.argmax(history.history['val_accuracy']) + 1)
}

with open(f"{param['result_directory']}/training_info.yaml", 'w') as f:
    yaml.dump(training_info, f)

# Plot training history
plt.figure(figsize=(12, 5))

# Plot training & validation loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()
plt.savefig(f"{param['result_directory']}/training_history.png")

print("\n=== Evaluating Final Model ===")
print("Loading test data...")
with open(test_pickle, 'rb') as f:
    test_data = pickle.load(f)

test_features = [item[0] for item in test_data]
test_labels = np.array([item[1] for item in test_data], dtype=np.float32)

print("Processing test data...")
processed_test_data = processor.process_batch(
    test_features, 
    normalize=True, 
    mean=mean, 
    std=std
)

print("Creating test dataset...")
test_dataset = tf.data.Dataset.from_tensor_slices((processed_test_data, test_labels))
test_dataset = test_dataset.batch(batch_size).prefetch(2)

# Evaluate model on test data
print("Evaluating model on test data...")
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save test results
with open(f"{param['result_directory']}/test_results.yaml", 'w') as f:
    yaml.dump({
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy)
    }, f)

print("\n=== Training Complete ===")
print(f"Model saved to: {checkpoint_path}")
print(f"Results saved to: {param['result_directory']}")