#!/usr/bin/env python3
"""
memory_optimized_ast_trainer.py - Memory-optimized script for AST model training with GPU
Designed to minimize GPU memory usage while maintaining model accuracy
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
import psutil  # For memory monitoring

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

# Force XLA compilation for better memory efficiency
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2'

# Memory release settings
os.environ['CUDA_CACHE_DISABLE'] = '1'

# Print system information
print("\n===== SYSTEM INFORMATION =====")
print("TensorFlow version:", tf.__version__)
import platform
print("Python version:", platform.python_version())
print("System:", platform.system(), platform.version())
print("CPU:", platform.processor())

# Memory monitoring function
def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_usage = mem_info.rss / (1024 * 1024 * 1024)  # GB
    print(f"RAM Usage: {ram_usage:.2f} GB")
    
    # GPU memory info if nvidia-smi is available
    try:
        gpu_memory = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits').read()
        print(f"GPU Memory Used: {gpu_memory.strip()} MB")
    except:
        pass
    
print_memory_usage()

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
        
        # We don't set hard memory limit to enable dynamic growth
        # tf.config.set_logical_device_configuration(
        #     physical_devices[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=8000)]  # 8GB limit
        # )
        # print("GPU memory limited to 8GB")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")

# Make sure directories exist
os.makedirs("./model/AST", exist_ok=True)
os.makedirs("./result/result_AST", exist_ok=True)
os.makedirs("./logs/log_AST", exist_ok=True)
os.makedirs("./temp_chunks/temp_chunks_AST", exist_ok=True)

# Load configuration
print("\n===== LOADING CONFIGURATION =====")
try:
    with open("baseline_AST.yaml", "r") as stream:
        param = yaml.safe_load(stream)
        print("Configuration loaded successfully")
except Exception as e:
    print(f"Error loading configuration: {e}")
    sys.exit(1)

# Apply memory optimizations to parameters
print("\n===== APPLYING MEMORY OPTIMIZATIONS =====")

# Reduce batch size to prevent OOM errors
original_batch_size = param['fit']['batch_size']
param['fit']['batch_size'] = 4  # Drastically reduced batch size
print(f"Batch size reduced: {original_batch_size} -> {param['fit']['batch_size']}")

# Increase gradient accumulation to compensate for smaller batch size
param['training']['gradient_accumulation_steps'] = 4
print(f"Gradient accumulation steps: {param['training']['gradient_accumulation_steps']}")

# Enable memory optimizations
param['training']['memory_optimization']['clear_memory_frequency'] = 2  # Clear memory more frequently
print(f"Memory clearing frequency: {param['training']['memory_optimization']['clear_memory_frequency']}")

# Enable gradient checkpointing (crucial for memory reduction)
param['training']['gradient_checkpointing']['enabled'] = True
print("Gradient checkpointing: Enabled")

# Reduce model complexity
param['model']['architecture']['transformer']['dim_feedforward'] = 64  # Reduced from default
param['model']['architecture']['transformer']['num_heads'] = 1  # Minimum heads
param['model']['architecture']['transformer']['num_encoder_layers'] = 1  # Minimum layers
print(f"Model complexity reduced - Feedforward: {param['model']['architecture']['transformer']['dim_feedforward']}, " +
      f"Heads: {param['model']['architecture']['transformer']['num_heads']}, " +
      f"Layers: {param['model']['architecture']['transformer']['num_encoder_layers']}")

# Use linear attention for memory efficiency
param['model']['architecture']['transformer']['attention_type'] = "linear"
print(f"Using memory-efficient attention: {param['model']['architecture']['transformer']['attention_type']}")

# Enable chunking to process dataset in smaller pieces
param['feature']['dataset_chunking']['enabled'] = True
param['feature']['dataset_chunking']['chunk_size'] = 500  # Small chunks
print(f"Dataset chunking: Enabled, Chunk size: {param['feature']['dataset_chunking']['chunk_size']}")

# Disable mixup for memory efficiency
param['training']['mixup']['enabled'] = False
print("Mixup augmentation: Disabled")

# Limit prefetch buffer size
param['training']['streaming_data']['prefetch_buffer_size'] = 1
param['training']['streaming_data']['parallel_calls'] = 2  # Reduced parallel operations
print(f"Prefetch buffer: {param['training']['streaming_data']['prefetch_buffer_size']}, " +
      f"Parallel calls: {param['training']['streaming_data']['parallel_calls']}")

# Enable mixed precision for faster training and lower memory usage
print("Enabling mixed precision...")
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed precision policy: {policy}")

# Enable XLA acceleration
print("Enabling XLA JIT compilation...")
tf.config.optimizer.set_jit(True)

# Function to check if we're close to OOM
def check_memory_status():
    try:
        gpu_memory_info = os.popen('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits').read().strip().split('\n')
        for info in gpu_memory_info:
            used, total = map(int, info.split(','))
            usage_percent = (used / total) * 100
            print(f"GPU Memory: {used}MB / {total}MB ({usage_percent:.1f}%)")
            if usage_percent > 90:
                print("WARNING: GPU memory usage above 90%! Consider reducing batch size further.")
                return True
        return False
    except:
        # If we can't check, assume we're fine
        return False

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

# Function to load pickle files efficiently
def load_pickle_efficiently(file_path, max_items=None):
    """Load pickle data efficiently, optionally loading just a subset to save memory"""
    print(f"Loading: {file_path}" + (" (partial)" if max_items else ""))
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if max_items and isinstance(data, (list, np.ndarray)) and len(data) > max_items:
                # Only keep the first max_items
                if isinstance(data, np.ndarray):
                    return data[:max_items].copy()
                else:
                    return data[:max_items]
            return data
    except Exception as e:
        print(f"Error loading pickle file {file_path}: {e}")
        return None

# Function to load large pickle files in chunks to prevent memory overflow
def load_pickle_in_chunks(file_path, chunk_size=1000):
    """Generator to load a pickle file in chunks to save memory"""
    print(f"Streaming data from: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        # Process data in chunks
        total_items = len(data)
        for i in range(0, total_items, chunk_size):
            end_idx = min(i + chunk_size, total_items)
            print(f"Loading chunk {i//chunk_size + 1}/{(total_items+chunk_size-1)//chunk_size}: items {i}-{end_idx-1}")
            
            if isinstance(data, np.ndarray):
                yield data[i:end_idx].copy()
            else:
                yield data[i:end_idx]
                
            # Force garbage collection
            gc.collect()

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

# Load a small batch of validation data for normalization statistics
print("\n===== PROCESSING SMALL VALIDATION SET FOR NORMALIZATION =====")
print("Loading small batch of validation data for normalization stats...")
val_data_sample = load_pickle_efficiently(val_pickle, max_items=1000)
val_labels_sample = load_pickle_efficiently(val_labels_pickle, max_items=1000)
print(f"Sample validation data shape: {val_data_sample.shape if hasattr(val_data_sample, 'shape') else 'N/A'}")

# Process validation data
print("Processing validation sample...")
processed_val_sample = np.zeros((len(val_data_sample), target_shape[0], target_shape[1]), dtype=np.float32)
for i in range(len(val_data_sample)):
    processed_val_sample[i] = preprocess_spectrogram(val_data_sample[i], target_shape)
    if i % 100 == 0 and i > 0:
        print(f"Processed {i}/{len(val_data_sample)} validation samples")
        # Check for potential OOM
        if check_memory_status():
            print("Reducing sample size due to memory pressure")
            processed_val_sample = processed_val_sample[:i]
            val_labels_sample = val_labels_sample[:i]
            break

print(f"Processed validation sample shape: {processed_val_sample.shape}")

# Calculate normalization statistics
mean = np.mean(processed_val_sample)
std = np.std(processed_val_sample)
print(f"Normalization statistics - Mean: {mean:.4f}, Std: {std:.4f}")

# Free memory
val_data_sample = None
gc.collect()
print_memory_usage()

# Create TensorFlow dataset for the validation sample
print("\n===== CREATING VALIDATION DATASET =====")
print("Creating validation dataset...")
val_labels_sample = np.array(val_labels_sample, dtype=np.float32)

# Apply normalization
if std > 0:
    processed_val_sample = (processed_val_sample - mean) / std
    print("Z-score normalization applied to validation data")
else:
    print("Warning: Standard deviation is 0, skipping normalization")

val_dataset = tf.data.Dataset.from_tensor_slices((processed_val_sample, val_labels_sample))
val_dataset = val_dataset.batch(
    param.get("fit", {}).get("batch_size", 4)
).prefetch(1)  # Minimal prefetch

# Free memory
processed_val_sample = None
val_labels_sample = None
gc.collect()
print_memory_usage()

# Define a memory-efficient data generator
class MemoryEfficientDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path, labels_path, batch_size, target_shape, mean, std, 
                 max_items_in_memory=100, shuffle=True):
        """
        A memory-efficient data generator that loads data in small chunks.
        
        Args:
            data_path: Path to data pickle file
            labels_path: Path to labels pickle file
            batch_size: Batch size for training
            target_shape: Target shape for spectrograms
            mean: Mean for normalization
            std: Standard deviation for normalization
            max_items_in_memory: Maximum number of items to keep in memory at once
            shuffle: Whether to shuffle the data
        """
        # Load all labels (they're usually small)
        self.labels = load_pickle_efficiently(labels_path)
        self.n_samples = len(self.labels)
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.mean = mean
        self.std = std
        self.max_items_in_memory = min(max_items_in_memory, self.n_samples)
        self.shuffle = shuffle
        
        # Current data chunk
        self.current_data_chunk = None
        self.current_chunk_start = 0
        self.current_chunk_end = 0
        
        # Create shuffled indices
        self.indices = np.random.permutation(self.n_samples) if shuffle else np.arange(self.n_samples)
        print(f"Created memory-efficient data generator with {self.n_samples} samples")
        print(f"Max items in memory: {self.max_items_in_memory}")
        
        # Pre-load first chunk
        self._load_next_chunk(0)
    
    def _load_next_chunk(self, start_idx):
        """Load the next chunk of data into memory"""
        # Clear current chunk from memory
        self.current_data_chunk = None
        gc.collect()
        
        # Determine chunk boundaries
        end_idx = min(start_idx + self.max_items_in_memory, self.n_samples)
        print(f"Loading data chunk {start_idx}-{end_idx-1} into memory")
        
        # Record chunk boundaries
        self.current_chunk_start = start_idx
        self.current_chunk_end = end_idx
        
        # Load relevant indices
        indices_to_load = self.indices[start_idx:end_idx]
        
        # For now, load full data then select needed indices
        # This is inefficient but pickle doesn't support partial loading
        with open(self.data_path, 'rb') as f:
            all_data = pickle.load(f)
            # Only keep the required indices
            self.current_data_chunk = [all_data[i] for i in indices_to_load]
            
        # Explicitly delete all_data to free memory
        all_data = None
        gc.collect()
        
        # Print memory usage after loading
        print_memory_usage()
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, idx):
        # Calculate which samples we need for this batch
        batch_start = idx * self.batch_size
        batch_end = min((idx + 1) * self.batch_size, self.n_samples)
        batch_indices = self.indices[batch_start:batch_end]
        
        # Initialize batch arrays
        batch_x = np.zeros((len(batch_indices), self.target_shape[0], self.target_shape[1]), dtype=np.float32)
        batch_y = np.array([self.labels[i] for i in batch_indices], dtype=np.float32)
        
        # Check if we need to load a new chunk
        for i, idx in enumerate(batch_indices):
            relative_idx = idx - self.current_chunk_start
            
            # Load next chunk if index out of current chunk bounds
            if idx < self.current_chunk_start or idx >= self.current_chunk_end:
                # Calculate which chunk we need
                chunk_start = (idx // self.max_items_in_memory) * self.max_items_in_memory
                self._load_next_chunk(chunk_start)
                
                # Recalculate relative index
                relative_idx = idx - self.current_chunk_start
            
            # Process data
            batch_x[i] = preprocess_spectrogram(
                self.current_data_chunk[relative_idx], 
                self.target_shape
            )
                    
        # Apply normalization
        if self.std > 0:
            batch_x = (batch_x - self.mean) / self.std
            
        return batch_x, batch_y
    
    def on_epoch_end(self):
        """Called at the end of every epoch"""
        # Shuffle indices if needed
        if self.shuffle:
            self.indices = np.random.permutation(self.n_samples)
            
        # Force reload of first chunk with new indices
        self._load_next_chunk(0)
        
        # Force garbage collection
        gc.collect()

# Enhanced version that implements gradient accumulation
class GradientAccumulationModel(tf.keras.Model):
    def __init__(self, *args, accumulation_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulation_steps = accumulation_steps
        self._gradient_accumulation_count = tf.Variable(0, trainable=False, dtype=tf.int32)
        self._accumulated_gradients = None
    
    def train_step(self, data):
        x, y = data
        
        # Reset accumulated gradients at the start of a new accumulation cycle
        if self._gradient_accumulation_count == 0:
            self._accumulated_gradients = None
            
        # Increment counter
        self._gradient_accumulation_count.assign_add(1)
        
        # Forward pass and calculate loss using GradientTape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            # Scale the loss to account for accumulation
            scaled_loss = loss / self.accumulation_steps
        
        # Calculate gradients
        gradients = tape.gradient(scaled_loss, self.trainable_variables)
        
        # Accumulate gradients
        if self._accumulated_gradients is None:
            self._accumulated_gradients = [tf.zeros_like(g) for g in gradients]
            
        for i, g in enumerate(gradients):
            if g is not None:
                self._accumulated_gradients[i] = self._accumulated_gradients[i] + g
        
        # Apply accumulated gradients when we've reached accumulation_steps
        if self._gradient_accumulation_count == self.accumulation_steps:
            self.optimizer.apply_gradients(zip(self._accumulated_gradients, self.trainable_variables))
            self._gradient_accumulation_count.assign(0)
            # Explicitly clear accumulated gradients to save memory
            self._accumulated_gradients = None
            
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        
        # Return a dict mapping metric names to current values
        return {m.name: m.result() for m in self.metrics}

# Function to create a positional encoding
def positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    pe = np.expand_dims(pe, axis=0)
    return tf.cast(pe, dtype=tf.float32)

# Memory-efficient AST Model (reduced complexity)
def create_memory_efficient_ast_model(input_shape, config=None):
    print(f"\n===== CREATING MEMORY-EFFICIENT MODEL =====")
    print(f"Creating lightweight AST model with input shape {input_shape}...")
    
    if config is None:
        config = {}
    
    # Get transformer configuration
    transformer_config = config.get("transformer", {})
    num_heads = transformer_config.get("num_heads", 1)  # Reduced from 4
    dim_feedforward = transformer_config.get("dim_feedforward", 64)  # Reduced from 128
    num_encoder_layers = transformer_config.get("num_encoder_layers", 1)  # Reduced from 4
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
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5)(x)
    
    # Patch embedding - simplified to reduce parameters
    x = tf.keras.layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding='valid',
        name='patch_embedding'
    )(x)
    
    # Reshape to sequence format
    x = tf.keras.layers.Reshape((seq_len, embed_dim))(x)
    
    # Layer normalization
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
    
    # Add positional encoding
    pos_encoding = positional_encoding(seq_len, embed_dim)
    x = tf.keras.layers.Add()([x, pos_encoding])
    
    # Apply transformer encoder layers with gradient checkpointing
    for i in range(num_encoder_layers):
        # Save the input for the residual connection
        residual = x
        
        # Apply layer norm before attention (Pre-LN helps training stability)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
        
        # Use linear attention for memory efficiency if specified
        attention_type = transformer_config.get("attention_type", "standard")
        if attention_type == "linear":
            # Linear attention implementation (more memory efficient)
            # Calculate queries, keys, values
            q = tf.keras.layers.Dense(embed_dim)(x)
            k = tf.keras.layers.Dense(embed_dim)(x)
            v = tf.keras.layers.Dense(embed_dim)(x)
            
            # Apply ELU activation to keys for positive values
            k = tf.keras.layers.Activation('elu')(k) + 1.0
            
            # Linear attention: (Q · (K^T · V))
            kv = tf.matmul(k, v, transpose_a=True)
            qkv = tf.matmul(q, kv)
            
            # Project back to embedding dimension
            attn_output = tf.keras.layers.Dense(embed_dim)(qkv)
        else:
            # Standard multi-head attention
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                dropout=attention_dropout
            )(x, x)
        
        # Apply dropout
        attn_output = tf.keras.layers.Dropout(0.1)(attn_output)
        
        # Add residual connection
        x = tf.keras.layers.Add()([residual, attn_output])
        
        # Save the input for the second residual connection
        residual = x
        
        # Apply layer normalization
        x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
        
        # Feed-forward network (reduced size)
        x = tf.keras.layers.Dense(embed_dim * 2, activation='gelu')(x)  # Reduced multiplier
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(embed_dim)(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        # Add second residual connection
        x = tf.keras.layers.Add()([residual, x])
    
    # Final layer normalization
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
    
    # Global average pooling to reduce memory
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Classification head (simplified)
    x = tf.keras.layers.Dense(embed_dim // 2, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    print(f"Model created with {model.count_params()} parameters")
    
    return model

# Define the periodic memory cleanup callback
class MemoryCleanupCallback(tf.keras.callbacks.Callback):
    def __init__(self, frequency=5):
        super().__init__()
        self.frequency = frequency
    
    def on_batch_end(self, batch, logs=None):
        if batch % self.frequency == 0:
            # Clear TensorFlow's GPU memory
            tf.keras.backend.clear_session()
            # Force Python garbage collection
            gc.collect()
            # Print an indicator that cleanup occurred
            print("🧹", end="", flush=True)

# Memory monitoring callback
class MemoryMonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, frequency=50):
        super().__init__()
        self.frequency = frequency
    
    def on_batch_end(self, batch, logs=None):
        if batch % self.frequency == 0:
            # Check for memory issues
            check_memory_status()

# Periodic model saving callback
class PeriodicModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, save_freq=100):
        super().__init__()
        self.save_dir = save_dir
        self.save_freq = save_freq
    
    def on_batch_end(self, batch, logs=None):
        if batch > 0 and batch % self.save_freq == 0:
            # Save intermediate model
            print(f"\nSaving intermediate model at batch {batch}")
            try:
                self.model.save_weights(f"{self.save_dir}/intermediate_batch_{batch}.keras")
                print("✓ Intermediate model saved")
            except Exception as e:
                print(f"✗ Failed to save intermediate model: {e}")

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

# Create model with gradient accumulation
print("\n===== CREATING AND COMPILING MODEL =====")
base_model = create_memory_efficient_ast_model(
    input_shape=target_shape, 
    config=param.get("model", {}).get("architecture", {})
)

# Wrap the model with gradient accumulation
model = GradientAccumulationModel(
    base_model.inputs,
    base_model.outputs,
    accumulation_steps=param.get("training", {}).get("gradient_accumulation_steps", 2)
)

print(f"Using gradient accumulation with {param.get('training', {}).get('gradient_accumulation_steps', 2)} steps")
print(f"Effective batch size: {param.get('fit', {}).get('batch_size', 4) * param.get('training', {}).get('gradient_accumulation_steps', 2)}")

# Compile model with a clipped optimizer to prevent exploding gradients
print("Compiling model...")
optimizer = AdamW(
    learning_rate=param.get("fit", {}).get("compile", {}).get("learning_rate", 0.0001),
    weight_decay=0.001,  # Reduced from 0.01 to prevent overtraining
    clipnorm=param.get("training", {}).get("gradient_clip_norm", 1.0)
)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define all callbacks
print("\n===== SETTING UP TRAINING =====")
print("Setting up training callbacks...")
callbacks = []

# Memory cleanup callback
callbacks.append(
    MemoryCleanupCallback(
        frequency=param.get("training", {}).get("memory_optimization", {}).get("clear_memory_frequency", 5)
    )
)

# Memory monitor callback
callbacks.append(MemoryMonitorCallback(frequency=50))

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
            factor=param.get("fit", {}).get("lr_scheduler", {}).get("factor", 0.5),  # Milder reduction
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

# Periodic model saving
callbacks.append(
    PeriodicModelSaver(
        save_dir=param['model_directory'],
        save_freq=param.get("training", {}).get("checkpointing", {}).get("save_frequency", 1000)
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

# TensorBoard logging (minimal)
log_dir = f"./logs/log_AST/tensorboard_{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
callbacks.append(
    tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,  # Disable histograms to save memory
        update_freq='epoch',  # Only update at the end of epochs
        profile_batch=0  # Disable profiling to save memory
    )
)

# GPU usage marker
class GPUUsageMarker(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 25 == 0:
            print("⚡", end="", flush=True)
            
callbacks.append(GPUUsageMarker())

# Create training data generator with memory efficiency
print("\nCreating memory-efficient training data generator...")
batch_size = param.get("fit", {}).get("batch_size", 4)
train_generator = MemoryEfficientDataGenerator(
    data_path=train_pickle,
    labels_path=train_labels_pickle,
    batch_size=batch_size,
    target_shape=target_shape,
    mean=mean,
    std=std,
    max_items_in_memory=100  # Only keep 100 samples in memory at a time
)

# Final GPU check
print("\n===== FINAL CHECKS =====")
print("GPU device name:", tf.test.gpu_device_name())

# Run a small test with the model
print("Testing model prediction...")
test_batch = np.random.random((batch_size, target_shape[0], target_shape[1])).astype(np.float32)
test_output = model.predict(test_batch, verbose=0)
print(f"Test prediction shape: {test_output.shape}")

# Get current memory usage
print("\nCurrent memory usage before training:")
print_memory_usage()

# Start training with try-except for better error handling
print("\n===== STARTING TRAINING =====")
print(f"Starting model training for {param['fit']['epochs']} epochs...")
print(f"Batch size: {batch_size}, Gradient accumulation steps: {param['training']['gradient_accumulation_steps']}")
print(f"Effective batch size: {batch_size * param['training']['gradient_accumulation_steps']}")
print("Training with memory-optimized configuration, mixed precision, and gradient accumulation")

# Begin timing
training_start_time = time.time()

try:
    # Train model
    history = model.fit(
        train_generator,
        epochs=param["fit"]["epochs"],
        validation_data=val_dataset,
        verbose=1,
        callbacks=callbacks,
        max_queue_size=2,  # Reduced queue size
        workers=2,  # Reduced workers
        use_multiprocessing=False  # Disable multiprocessing to save memory
    )

    training_successful = True
    
except Exception as e:
    print(f"\n\nERROR DURING TRAINING: {e}")
    print("\nAttempting to save partial model...")
    training_successful = False
    
    # Try to extract history
    try:
        history = model.history
    except:
        history = None

# End timing
training_end_time = time.time()
training_duration = training_end_time - training_start_time
print(f"\nTraining {'completed' if training_successful else 'interrupted'} after {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")

# Save trained model (even if training was interrupted)
try:
    # Save model in the native Keras format
    model_path = f"{param['model_directory']}/{'final' if training_successful else 'partial'}_model"
    model.save(f"{model_path}.keras", save_format='keras')
    print(f"Model saved successfully as {model_path}.keras")
    
except Exception as e:
    print(f"Error saving full model: {e}")
    
    # Try to save just the weights
    try:
        print("Attempting to save weights only...")
        model.save_weights(f"{model_path}_weights.keras")
        print("Model weights saved successfully")
    except Exception as e2:
        print(f"Error saving weights: {e2}")

# Explicitly clean up to free GPU memory
tf.keras.backend.clear_session()
gc.collect()
print("\nFinal memory usage after training:")
print_memory_usage()

# Plot training history if available
if history and hasattr(history, 'history') and history.history:
    print("Plotting training history...")
    plt.figure(figsize=(12, 8))
    
    metrics_to_plot = [
        ('loss', 'val_loss', 'Model loss'),
        ('accuracy', 'val_accuracy', 'Model accuracy')
    ]
    
    for i, (train_metric, val_metric, title) in enumerate(metrics_to_plot, 1):
        if train_metric in history.history and val_metric in history.history:
            plt.subplot(2, 1, i)
            plt.plot(history.history[train_metric])
            plt.plot(history.history[val_metric])
            plt.title(title)
            plt.ylabel(train_metric.capitalize())
            plt.xlabel("Epoch")
            plt.legend(["Train", "Validation"], loc="best")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{param['result_directory']}/training_history.png")
    plt.close()
    print(f"Training history plot saved to: {param['result_directory']}/training_history.png")

print("\n===== TRAINING COMPLETE =====")
print(f"Memory-optimized training {'completed successfully' if training_successful else 'was interrupted but partial model was saved'}!")
print(f"Model saved to: {model_path}.keras")

# Print memory usage tips
print("\n===== MEMORY OPTIMIZATION TIPS =====")
print("1. If you're still experiencing OOM errors, try further reducing the batch size")
print("2. Increase gradient accumulation steps to maintain effective batch size")
print("3. Further reduce model complexity (num_heads, dim_feedforward, num_encoder_layers)")
print("4. Use linear attention mode for better memory efficiency")
print("5. Enable chunking for dataset processing")
print("6. Make sure you have no other applications using GPU memory")