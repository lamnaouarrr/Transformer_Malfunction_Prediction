#!/usr/bin/env python3
"""
gpu_optimized_trainer.py - A high-performance GPU-optimized trainer for AST model
This script ensures GPU computation is properly utilized and memory is efficiently managed
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
import gc  # For garbage collection

# Force TensorFlow to use the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # Use async allocator to reduce fragmentation

# Clear any existing TensorFlow session and free GPU memory
print("Clearing GPU memory...")
tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth enabled for {len(gpus)} GPUs")
        
        # Only allocate GPU memory as needed
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)]  # 20GB limit
        )
        print("GPU memory limited to 20GB")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")

# Enable XLA for faster training
print("Enabling XLA JIT compilation...")
tf.config.optimizer.set_jit(True)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2'

# Make sure directories exist
os.makedirs("./model/AST", exist_ok=True)
os.makedirs("./result/result_AST", exist_ok=True)
os.makedirs("./logs/log_AST", exist_ok=True)

# Load configuration
print("Loading configuration...")
try:
    with open("baseline_AST.yaml", "r") as stream:
        param = yaml.safe_load(stream)
        print("Configuration loaded successfully")
except Exception as e:
    print(f"Error loading configuration: {e}")
    sys.exit(1)

# Enable mixed precision for faster training
print("Enabling mixed precision...")
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed precision policy: {policy}")

# Define paths to pickle files
pickle_dir = param['pickle_directory']
train_pickle = f"{pickle_dir}/train_overall.pickle"
train_labels_pickle = f"{pickle_dir}/train_labels_overall.pickle"
val_pickle = f"{pickle_dir}/val_overall.pickle"
val_labels_pickle = f"{pickle_dir}/val_labels_overall.pickle"
test_files_pickle = f"{pickle_dir}/test_files_overall.pickle"
test_labels_pickle = f"{pickle_dir}/test_labels_overall.pickle"

print(f"Checking pickle files in: {pickle_dir}")

# Check if all files exist
for p in [train_pickle, train_labels_pickle, val_pickle, val_labels_pickle, test_files_pickle, test_labels_pickle]:
    if os.path.exists(p):
        print(f"Found: {p} ({os.path.getsize(p) / (1024 * 1024):.2f} MB)")
    else:
        print(f"Missing: {p}")
        sys.exit(1)

# Function to load pickle files
def load_pickle(file_path):
    print(f"Loading: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Function to preprocess a single spectrogram to ensure consistent shape
def preprocess_spectrogram(spec, target_shape):
    # Handle 3D input
    if len(spec.shape) == 3 and spec.shape[2] == 1:
        spec = spec[:, :, 0]
    
    # Skip if dimensions already match
    if spec.shape[0] == target_shape[0] and spec.shape[1] == target_shape[1]:
        return spec
        
    # Simple resize using zero padding/cropping
    temp_spec = np.zeros(target_shape, dtype=np.float32)
    freq_dim = min(spec.shape[0], target_shape[0])
    time_dim = min(spec.shape[1], target_shape[1])
    temp_spec[:freq_dim, :time_dim] = spec[:freq_dim, :time_dim]
    return temp_spec

# Define target shape for spectrograms
target_shape = (param["feature"]["n_mels"], 96)
print(f"Target spectrogram shape: {target_shape}")

# Enhanced batch processing with parallel computations
class FastSpecDataProcessor:
    def __init__(self, target_shape):
        self.target_shape = target_shape
        
    def process_batch(self, batch_data, normalize=False, mean=0, std=1):
        processed = np.zeros((len(batch_data), self.target_shape[0], self.target_shape[1]), dtype=np.float32)
        
        for i, spec in enumerate(batch_data):
            processed[i] = preprocess_spectrogram(spec, self.target_shape)
            
        if normalize and std > 0:
            processed = (processed - mean) / std
            
        return processed

# Create data processor
processor = FastSpecDataProcessor(target_shape)

# Load validation data in one go (it's small enough)
print("\nLoading validation data...")
val_data = load_pickle(val_pickle)
val_labels = load_pickle(val_labels_pickle)
print(f"Validation data shape: {val_data.shape}")
print(f"Validation labels shape: {val_labels.shape}")

# Get normalization statistics from a sample of training data
print("\nCalculating normalization statistics...")
with open(train_pickle, 'rb') as f:
    train_sample = pickle.load(f)[:2000]  # Sample first 2000 for statistics
    
processed_sample = processor.process_batch(train_sample)
mean = np.mean(processed_sample)
std = np.std(processed_sample)
print(f"Training statistics - Mean: {mean:.4f}, Std: {std:.4f}")

# Free memory
train_sample = None
processed_sample = None
gc.collect()

# Process all validation data at once
print("Processing validation data...")
processed_val_data = processor.process_batch(val_data, normalize=True, mean=mean, std=std)
    
# Free memory
val_data = None
gc.collect()

print(f"Processed validation data shape: {processed_val_data.shape}")

# Create validation dataset with prefetching and caching for speed
print("Creating optimized validation dataset...")
val_dataset = tf.data.Dataset.from_tensor_slices((processed_val_data, val_labels))
val_dataset = val_dataset.cache().batch(
    param.get("fit", {}).get("batch_size", 16)
).prefetch(tf.data.AUTOTUNE)

# Free memory
processed_val_data = None
gc.collect()

# Function to create a positional encoding
def positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    pe = np.expand_dims(pe, axis=0)
    return tf.cast(pe, dtype=tf.float32)

# Function to create the AST model with speed optimizations
def create_ast_model(input_shape, config=None):
    print(f"Creating optimized AST model with input shape {input_shape}...")
    if config is None:
        config = {}
    
    # Get transformer configuration
    transformer_config = config.get("transformer", {})
    num_heads = transformer_config.get("num_heads", 1)
    dim_feedforward = transformer_config.get("dim_feedforward", 128)
    num_encoder_layers = transformer_config.get("num_encoder_layers", 1)
    patch_size = transformer_config.get("patch_size", 4)
    attention_dropout = transformer_config.get("attention_dropout", 0.1)
    
    # Calculate sequence length and embedding dimension
    h_patches = input_shape[0] // patch_size
    w_patches = input_shape[1] // patch_size
    seq_len = h_patches * w_patches
    embed_dim = dim_feedforward
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Add channel dimension
    x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    # Add batch normalization
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Patch embedding
    x = tf.keras.layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding='valid',
        name='patch_embedding'
    )(x)
    
    # Reshape to sequence format
    x = tf.keras.layers.Reshape((seq_len, embed_dim))(x)
    
    # Layer normalization before adding positional encoding
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Add positional encoding
    pos_encoding = positional_encoding(seq_len, embed_dim)
    x = tf.keras.layers.Add()([x, pos_encoding])
    
    # Apply transformer encoder layers
    for i in range(num_encoder_layers):
        # Multi-head attention
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
        
        # Feed-forward network
        ffn = tf.keras.layers.Dense(embed_dim * 4, activation='gelu')(x)
        ffn = tf.keras.layers.Dense(embed_dim)(ffn)
        
        # Add dropout and residual connection
        ffn = tf.keras.layers.Dropout(0.1)(ffn)
        x = tf.keras.layers.Add()([x, ffn])
        
        # Layer normalization
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = tf.keras.layers.Dense(embed_dim // 2, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    print(f"Model created with {model.count_params()} parameters")
    return model

# Create a high-performance data generator
class HighPerformanceDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path, labels_path, batch_size, target_shape, mean, std, buffer_size=100):
        # Load all labels (they're small)
        self.labels = load_pickle(labels_path)
        self.n_samples = len(self.labels)
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.mean = mean
        self.std = std
        self.buffer_size = min(buffer_size, self.n_samples)  # Don't buffer more than we have
        
        # Pre-load some data into buffer for faster access
        print(f"Preloading {self.buffer_size} samples into memory buffer...")
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
            # Only keep the buffer_size amount in memory
            self.data_buffer = self.data[:self.buffer_size]
            if len(self.data) > self.buffer_size:
                self.data = None  # Free full data from memory if we're only using the buffer
                
        # Create shuffled indices
        self.indices = np.random.permutation(self.n_samples)
        
        print(f"Created high-performance generator with {self.n_samples} samples")
        self.processor = FastSpecDataProcessor(target_shape)
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, idx):
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = np.array([self.labels[i] for i in batch_indices], dtype=np.float32)
        
        # Check which indices we need to load from disk vs from buffer
        buffer_indices = []
        disk_indices = []
        
        for i, idx in enumerate(batch_indices):
            if idx < self.buffer_size:
                buffer_indices.append((i, idx))
            else:
                disk_indices.append((i, idx))
                
        # Initialize batch arrays
        batch_x = np.zeros((len(batch_indices), self.target_shape[0], self.target_shape[1]), dtype=np.float32)
        
        # Fill from buffer (fast)
        for batch_idx, data_idx in buffer_indices:
            batch_x[batch_idx] = preprocess_spectrogram(self.data_buffer[data_idx], self.target_shape)
            
        # Fill from disk (slow, only if needed)
        if disk_indices:
            # If we need to load from disk, we need the full data
            if self.data is None:
                with open(self.data_path, 'rb') as f:
                    full_data = pickle.load(f)
                    for batch_idx, data_idx in disk_indices:
                        batch_x[batch_idx] = preprocess_spectrogram(full_data[data_idx], self.target_shape)
            else:
                # We already have all data in memory
                for batch_idx, data_idx in disk_indices:
                    batch_x[batch_idx] = preprocess_spectrogram(self.data[data_idx], self.target_shape)
                    
        # Apply normalization
        if self.std > 0:
            batch_x = (batch_x - self.mean) / self.std
            
        return batch_x, batch_y
    
    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch
        self.indices = np.random.permutation(self.n_samples)

# Print TensorFlow device placement to verify GPU usage
print("\nGPU Information:")
print("TensorFlow version:", tf.__version__)
print("Is GPU available:", tf.config.list_physical_devices('GPU'))
print("Is built with CUDA:", tf.test.is_built_with_cuda())
print("Testing GPU:")
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print("Matrix multiplication result shape:", c.shape)
    print("Device: ", a.device, b.device, c.device)

# Create model
model = create_ast_model(
    input_shape=target_shape, 
    config=param.get("model", {}).get("architecture", {})
)

# Compile model with speed optimizations
print("Compiling optimized model...")
optimizer = AdamW(
    learning_rate=param.get("fit", {}).get("compile", {}).get("learning_rate", 0.0001),
    weight_decay=0.01,
    clipnorm=param.get("training", {}).get("gradient_clip_norm", 1.0)
)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
print("Setting up training callbacks...")
callbacks = []

# Early stopping
if param.get("fit", {}).get("early_stopping", {}).get("enabled", True):
    print("Adding early stopping callback")
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=param.get("fit", {}).get("early_stopping", {}).get("monitor", "val_loss"),
            patience=param.get("fit", {}).get("early_stopping", {}).get("patience", 10),
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

# Model checkpointing
if param.get("fit", {}).get("checkpointing", {}).get("enabled", True):
    print("Adding model checkpoint callback")
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{param['model_directory']}/best_model.keras",
            monitor=param.get("fit", {}).get("checkpointing", {}).get("monitor", "val_accuracy"),
            save_best_only=True,
            mode=param.get("fit", {}).get("checkpointing", {}).get("mode", "max")
        )
    )

# Progress tracking and logging
callbacks.append(
    tf.keras.callbacks.CSVLogger(
        f"{param['logging']['file']}",
        separator=',',
        append=False
    )
)

# Memory cleanup callback
class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Clean up every 5 epochs
            print("\nPerforming memory cleanup...")
            gc.collect()
            tf.keras.backend.clear_session()
            print("Memory cleanup complete.")

callbacks.append(MemoryCleanupCallback())

# Create training data generator
print("Creating high-performance training data generator...")
batch_size = param.get("fit", {}).get("batch_size", 16)
train_generator = HighPerformanceDataGenerator(
    data_path=train_pickle,
    labels_path=train_labels_pickle,
    batch_size=batch_size,
    target_shape=target_shape,
    mean=mean,
    std=std,
    buffer_size=1000  # Store 1000 samples in memory for faster access
)

# Train model with speed optimizations
print(f"\nStarting high-performance model training for {param['fit']['epochs']} epochs...")
print(f"Batch size: {batch_size}, Optimizer: {param['fit']['compile']['optimizer']}")
print("Training will use GPU acceleration with mixed precision")

training_start_time = time.time()

history = model.fit(
    train_generator,
    epochs=param["fit"]["epochs"],
    validation_data=val_dataset,
    verbose=1,
    callbacks=callbacks,
    workers=4,  # Increase worker threads for data loading
    use_multiprocessing=True,  # Use multiprocessing for data loading
    max_queue_size=10  # Increase queue size for smoother batch delivery
)

training_end_time = time.time()
training_duration = training_end_time - training_start_time
print(f"\nTraining completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")

# Save trained model
try:
    model.save(f"{param['model_directory']}/final_model.keras")
    print("Model saved successfully")
except Exception as e:
    print(f"Error saving model: {e}")

# Plot training history
print("Plotting training history...")
plt.figure(figsize=(12, 8))

# Plot loss
plt.subplot(2, 1, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper right")

# Plot accuracy
plt.subplot(2, 1, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="lower right")

# Save figure
plt.tight_layout()
plt.savefig(f"{param['result_directory']}/training_history.png")
plt.close()

print("\nHigh-performance training script completed successfully!")