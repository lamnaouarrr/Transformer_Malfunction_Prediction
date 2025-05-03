#!/usr/bin/env python3
"""
gpu_trainer.py - A fixed script that properly utilizes GPU computation for AST model training
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

# Clear any existing TF sessions
tf.keras.backend.clear_session()

# Force TensorFlow to use the GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Add explicit memory cleanup settings
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'

# Force XLA compilation
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2'

# Add explicit GPU memory release settings for CUDA
os.environ['CUDA_CACHE_DISABLE'] = '1'

# Print system information
print("\n===== SYSTEM INFORMATION =====")
print("TensorFlow version:", tf.__version__)
import platform
print("Python version:", platform.python_version())
print("System:", platform.system(), platform.version())
print("CPU:", platform.processor())

# Check GPU availability
print("\n===== GPU CONFIGURATION =====")
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", len(physical_devices))
for i, gpu in enumerate(physical_devices):
    print(f"GPU {i}: {gpu}")

if not physical_devices:
    print("ERROR: No GPU detected! Training will be extremely slow.")
    print("Please make sure GPU drivers are properly installed.")
    user_input = input("Do you want to continue with CPU training? (y/n): ")
    if user_input.lower() != 'y':
        sys.exit(1)
else:
    # Configure GPU memory growth for all available GPUs
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for all GPUs")
        
        # Set memory limit but don't limit too aggressively
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=28000)]  # 28GB limit
        )
        print("GPU memory limited to 28GB")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")

# Make sure directories exist
os.makedirs("./model/AST", exist_ok=True)
os.makedirs("./result/result_AST", exist_ok=True)
os.makedirs("./logs/log_AST", exist_ok=True)

# Load configuration
print("\n===== LOADING CONFIGURATION =====")
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

# Enable XLA acceleration
print("Enabling XLA JIT compilation...")
tf.config.optimizer.set_jit(True)

# Define paths to pickle files
pickle_dir = param['pickle_directory']
train_pickle = f"{pickle_dir}/train_overall.pickle"
train_labels_pickle = f"{pickle_dir}/train_labels_overall.pickle"
val_pickle = f"{pickle_dir}/val_overall.pickle"
val_labels_pickle = f"{pickle_dir}/val_labels_overall.pickle"
test_files_pickle = f"{pickle_dir}/test_files_overall.pickle"
test_labels_pickle = f"{pickle_dir}/test_labels_overall.pickle"

print("\n===== CHECKING FILES =====")
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
print(f"\nTarget spectrogram shape: {target_shape}")

# Load and process validation data
print("\n===== PROCESSING VALIDATION DATA =====")
print("Loading validation data...")
val_data = load_pickle(val_pickle)
val_labels = load_pickle(val_labels_pickle)
print(f"Validation data shape: {val_data.shape}")
print(f"Validation labels shape: {val_labels.shape}")

# Process validation data
print("Processing validation data...")
processed_val_data = np.zeros((val_data.shape[0], target_shape[0], target_shape[1]), dtype=np.float32)
for i in range(val_data.shape[0]):
    processed_val_data[i] = preprocess_spectrogram(val_data[i], target_shape)
    if i % 500 == 0 and i > 0:
        print(f"Processed {i}/{val_data.shape[0]} validation samples")
print(f"Processed validation data shape: {processed_val_data.shape}")

# Calculate normalization statistics
mean = np.mean(processed_val_data)
std = np.std(processed_val_data)
print(f"Normalization statistics - Mean: {mean:.4f}, Std: {std:.4f}")

# Apply normalization
if std > 0:
    processed_val_data = (processed_val_data - mean) / std
    print("Z-score normalization applied to validation data")
else:
    print("Warning: Standard deviation is 0, skipping normalization")

# Free memory
val_data = None
gc.collect()

# Convert labels to float32
val_labels = np.array(val_labels, dtype=np.float32)

# Create TensorFlow dataset
print("\n===== CREATING DATASETS =====")
print("Creating validation dataset...")
val_dataset = tf.data.Dataset.from_tensor_slices((processed_val_data, val_labels))
val_dataset = val_dataset.batch(
    param.get("fit", {}).get("batch_size", 16)
).cache().prefetch(tf.data.AUTOTUNE)

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

# Function to create the AST model
def create_ast_model(input_shape, config=None):
    print(f"\n===== CREATING MODEL =====")
    print(f"Creating AST model with input shape {input_shape}...")
    
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

# Create an enhanced data generator that loads data efficiently
class EffectiveDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path, labels_path, batch_size, target_shape, mean, std, buffer_size=1000):
        # Load all labels (they're small)
        self.labels = load_pickle(labels_path)
        self.n_samples = len(self.labels)
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.mean = mean
        self.std = std
        self.buffer_size = min(buffer_size, self.n_samples)
        
        # Pre-load some data into buffer for faster access
        print(f"Preloading {self.buffer_size} samples into memory buffer...")
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
            # Only keep buffer_size amount in memory
            self.data_buffer = self.data[:self.buffer_size]
            if len(self.data) > self.buffer_size:
                self.data = None  # Free full data from memory
                
        # Create shuffled indices
        self.indices = np.random.permutation(self.n_samples)
        print(f"Created data generator with {self.n_samples} samples")
    
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

# Verify GPU usage
print("\n===== VERIFYING GPU USAGE =====")
print("Running test computation on GPU...")

# This is the critical part - check if TensorFlow actually using the GPU
try:
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        
        # Time the matrix multiplication
        start_time = time.time()
        c = tf.matmul(a, b)
        
        # Force execution
        c_value = c.numpy()
        end_time = time.time()
        
        print(f"GPU matrix multiplication time: {end_time - start_time:.4f} seconds")
        print(f"Matrix on device: {a.device}")
        print(f"Computation on device: {c.device}")
        
        if 'GPU:0' in str(c.device):
            print("GPU computation successful!")
        else:
            print("WARNING: Computation may not be using GPU!")
except Exception as e:
    print(f"ERROR during GPU test: {e}")
    print("Continuing with CPU fallback...")

# Create model
print("\n===== CREATING AND COMPILING MODEL =====")
model = create_ast_model(
    input_shape=target_shape, 
    config=param.get("model", {}).get("architecture", {})
)

# Compile model
print("Compiling model...")
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
print("\n===== SETTING UP TRAINING =====")
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

# Logging
callbacks.append(
    tf.keras.callbacks.CSVLogger(
        f"{param['logging']['file']}",
        separator=',',
        append=False
    )
)

# TensorBoard logging
log_dir = f"./logs/log_AST/tensorboard_{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
callbacks.append(
    tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
)

# GPU usage marker
class GPUUsageMarker(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 50 == 0:
            print("⚡", end="", flush=True)
            
callbacks.append(GPUUsageMarker())

# Create training data generator
print("\nCreating training data generator...")
batch_size = param.get("fit", {}).get("batch_size", 16)
train_generator = EffectiveDataGenerator(
    data_path=train_pickle,
    labels_path=train_labels_pickle,
    batch_size=batch_size,
    target_shape=target_shape,
    mean=mean,
    std=std,
    buffer_size=1000  # Store 1000 samples in memory for faster access
)

# Final GPU check
print("\n===== FINAL CHECKS =====")
print("GPU device name:", tf.test.gpu_device_name())

# Run a small test with the model
print("Testing model prediction...")
test_batch = np.random.random((batch_size, target_shape[0], target_shape[1])).astype(np.float32)
test_output = model.predict(test_batch, verbose=0)
print(f"Test prediction shape: {test_output.shape}")

# Start training
print("\n===== STARTING TRAINING =====")
print(f"Starting model training for {param['fit']['epochs']} epochs...")
print(f"Batch size: {batch_size}")
print("Training with GPU acceleration and mixed precision")

# Begin timing
training_start_time = time.time()

# Train model
history = model.fit(
    train_generator,
    epochs=param["fit"]["epochs"],
    validation_data=val_dataset,
    verbose=1,
    callbacks=callbacks,
    max_queue_size=20,
    workers=8,
    use_multiprocessing=True
)

# End timing
training_end_time = time.time()
training_duration = training_end_time - training_start_time
print(f"\nTraining completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")

# Save trained model
try:
    # Use save_weights instead of save to avoid the 'options' parameter issue
    model_path = f"{param['model_directory']}/final_model"
    model.save_weights(f"{model_path}.weights.h5")
    
    # Save model architecture separately as JSON
    model_json = model.to_json()
    with open(f"{model_path}.json", "w") as json_file:
        json_file.write(model_json)
        
    print("Model saved successfully (architecture and weights separately)")
    
    # Explicitly clean up to free GPU memory
    tf.keras.backend.clear_session()
    
    # Force garbage collection
    gc.collect()
    
    # Attempt to explicitly release GPU memory
    if physical_devices:
        try:
            import ctypes
            libgpuarray = ctypes.CDLL("libgpuarray.so")
            libgpuarray.gpu_clean_up()
            print("Explicitly released GPU memory")
        except:
            print("Could not explicitly release GPU memory via libgpuarray")
    
except Exception as e:
    print(f"Error saving model: {e}")
    
    # Try alternative saving method
    try:
        print("Attempting alternative saving method...")
        model.save(f"{param['model_directory']}/final_model", save_format='tf')
        print("Model saved with TensorFlow format")
    except Exception as e2:
        print(f"Alternative saving method also failed: {e2}")

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

print("\n===== TRAINING COMPLETE =====")
print("GPU-accelerated training completed successfully!")
print(f"Model saved to: {param['model_directory']}/final_model.keras")
print(f"Training history plot saved to: {param['result_directory']}/training_history.png")

# Add final cleanup function
def release_gpu_memory():
    print("\n===== RELEASING GPU MEMORY =====")
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    
    # Force garbage collection
    gc.collect()
    
    # Free CPU memory
    import sys
    for name in dir():
        if not name.startswith('_'):
            del globals()[name]
    gc.collect()
    
    # Try to manually release CUDA memory
    try:
        print("Attempting to run CUDA memory cleanup...")
        from numba import cuda
        cuda.select_device(0)
        cuda.close()
        print("CUDA memory cleanup successful")
    except:
        print("Could not explicitly release CUDA memory")
        
    # Use Linux process cleanup
    try:
        import os
        print("Attempting OS-level memory sync...")
        os.system('sync')
        print("OS memory sync complete")
    except:
        pass
    
    print("GPU memory cleanup completed. If memory remains allocated, try manually restarting your Python kernel.")

# Release GPU memory before exit
release_gpu_memory()