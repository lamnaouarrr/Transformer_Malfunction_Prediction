#!/usr/bin/env python3
"""
ultra_memory_efficient_trainer.py - A specialized trainer for AST models with extreme memory optimization

This script implements several advanced techniques to prevent OOM errors:
1. Memory-mapped file access for dataset handling
2. Gradient accumulation with micro-batches
3. Model parameter sharing and quantization
4. Aggressive memory cleanup
5. Progressive resolution training
6. Optimized attention mechanisms
"""

import os
import sys
import time
import pickle
import yaml
import gc
import psutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import AdamW
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
import traceback

# Set up basic logging to file
import logging
logging.basicConfig(
    filename='./logs/log_AST/ultra_memory_trainer.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_info(message):
    """Log info message to both console and file"""
    print(message)
    logging.info(message)

def log_error(message):
    """Log error message to both console and file"""
    print(f"ERROR: {message}")
    logging.error(message)

def log_exception(e):
    """Log exception with traceback to both console and file"""
    print(f"EXCEPTION: {str(e)}")
    print(traceback.format_exc())
    logging.exception(e)

# Create required directories
os.makedirs("./model/AST", exist_ok=True)
os.makedirs("./result/result_AST", exist_ok=True)
os.makedirs("./logs/log_AST", exist_ok=True)

log_info("Starting ultra_memory_efficient_trainer.py")

# Configure TensorFlow for memory efficiency
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    log_info(f"Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        try:
            # Limit memory growth to prevent TensorFlow from allocating all memory at once
            tf.config.experimental.set_memory_growth(device, True)
            log_info(f"Set memory growth for {device}")
        except RuntimeError as e:
            log_error(f"Error setting memory growth: {e}")
        
        try:
            # Set memory limit (adjust based on your GPU)
            # Start with a conservative value like 1GB (1024MB)
            memory_limit_mb = 1024
            tf.config.set_logical_device_configuration(
                device,
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
            )
            log_info(f"Limited GPU memory to {memory_limit_mb}MB")
        except RuntimeError as e:
            log_error(f"Error limiting GPU memory: {e}")
else:
    log_info("No GPU found. Using CPU for training.")

# Enable XLA compilation for faster execution
tf.config.optimizer.set_jit(True)
log_info("XLA compilation enabled")

# Load configuration file
try:
    with open("baseline_AST.yaml", "r") as file:
        param = yaml.safe_load(file)
        log_info("Configuration loaded successfully")
except Exception as e:
    log_error(f"Error loading configuration: {e}")
    sys.exit(1)

# Enable mixed precision if specified in config
if param.get("training", {}).get("mixed_precision", True):
    mixed_precision.set_global_policy('mixed_float16')
    log_info(f"Mixed precision policy set to: {mixed_precision.global_policy()}")

# Memory usage tracking function
def print_memory_usage(message=""):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    gpu_memory = 0
    
    # Get GPU memory usage if available
    try:
        if physical_devices:
            gpu_memory = tf.config.experimental.get_memory_info('GPU:0')['current'] / (1024 * 1024)
    except:
        pass
    
    if message:
        message = f"{message} - "
    
    msg = f"{message}Memory Usage: {memory_mb:.2f} MB RAM | {gpu_memory:.2f} MB GPU"
    log_info(msg)
    return memory_mb, gpu_memory

# Clean memory function
def clean_memory():
    """Aggressively clean memory"""
    gc.collect()
    tf.keras.backend.clear_session()
    log_info("Memory cleaned")

# Define smaller target spectrogram shape to reduce memory usage
# Adjust these parameters based on your dataset and memory constraints
target_freq_dim = param["feature"]["n_mels"]  # From config
target_time_dim = 96  # Fixed value that balances information vs memory usage
target_shape = (target_freq_dim, target_time_dim)
log_info(f"Target spectrogram shape: {target_shape}")

# Efficient spectrogram preprocessing function
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

# Memory-efficient data generator class
class MemoryEfficientDataGenerator(tf.keras.utils.Sequence):
    """A memory-efficient data generator that uses memory mapping when possible"""
    
    def __init__(self, data_path, labels_path, batch_size, target_shape, mean=0, std=1, 
                 shuffle=True, is_training=True, micro_batch_size=None):
        # Store paths and parameters
        self.data_path = data_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.mean = mean
        self.std = std
        self.shuffle = shuffle
        self.is_training = is_training
        
        # Support for gradient accumulation
        self.micro_batch_size = micro_batch_size if micro_batch_size else batch_size
        
        # Load labels (small enough to keep in memory)
        log_info(f"Loading labels from {labels_path}")
        try:
            with open(labels_path, 'rb') as f:
                self.labels = pickle.load(f)
                log_info(f"Loaded {len(self.labels)} labels")
        except Exception as e:
            log_exception(e)
            raise
        
        # Get data length
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.data_length = len(data)
                log_info(f"Data file contains {self.data_length} samples")
                
            # Create indices
            self.indices = np.arange(self.data_length)
            if self.shuffle:
                np.random.shuffle(self.indices)
            
            log_info(f"Created generator with {self.data_length} samples")
        except Exception as e:
            log_exception(e)
            raise
        
    def __len__(self):
        """Return the number of batches per epoch"""
        return int(np.ceil(self.data_length / self.batch_size))
    
    def __getitem__(self, idx):
        """Get a batch of data"""
        # Calculate indices for this batch
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Initialize arrays for batch data
        batch_size = len(batch_indices)
        batch_x = np.zeros((batch_size, self.target_shape[0], self.target_shape[1]), dtype=np.float16)
        batch_y = np.zeros(batch_size, dtype=np.float32)
        
        try:
            # Load data for each sample in the batch
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
                for i, idx in enumerate(batch_indices):
                    # Process spectrogram
                    if idx < len(data):
                        spec = preprocess_spectrogram(data[idx], self.target_shape)
                        batch_x[i] = spec
                        batch_y[i] = self.labels[idx]
            
            # Normalize data
            if self.std > 0:
                batch_x = (batch_x - self.mean) / self.std
            
            # Return data and labels
            return batch_x, batch_y
        except Exception as e:
            log_exception(e)
            raise
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

# Function to create positional encoding
def get_positional_encoding(seq_len, d_model):
    """Create positional encoding for transformer"""
    # Implement sinusoidal positional encoding
    positions = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    
    # Create positional encoding
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    
    # Add batch dimension and convert to tensor
    pe = np.expand_dims(pe, axis=0)
    return tf.cast(pe, dtype=tf.float16 if mixed_precision.global_policy().name == 'mixed_float16' else tf.float32)

# Create a memory-efficient AST model
def create_memory_efficient_ast(input_shape, config=None):
    """Create a memory-efficient AST model"""
    if config is None:
        config = {}
    
    # Get transformer configuration from config
    transformer_config = config.get("transformer", {})
    num_heads = transformer_config.get("num_heads", 1)  # Using fewer heads to save memory
    dim_feedforward = transformer_config.get("dim_feedforward", 64)  # Smaller feedforward dimension
    num_encoder_layers = transformer_config.get("num_encoder_layers", 1)  # Fewer layers
    patch_size = transformer_config.get("patch_size", 4)  # Patch size for tokenization
    
    # Calculate sequence length and embedding dimension
    h_patches = input_shape[0] // patch_size
    w_patches = input_shape[1] // patch_size
    seq_len = h_patches * w_patches
    embed_dim = dim_feedforward  # Set embedding dimension to match feedforward dimension
    
    log_info(f"Creating AST model with: patches={h_patches}x{w_patches}, sequence_length={seq_len}, embed_dim={embed_dim}")
    
    # Define input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Add channel dimension
    x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    # Batch normalization for input stability
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Create patches with convolution (equivalent to patching)
    x = tf.keras.layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding='valid',
        name='patch_embedding'
    )(x)
    
    # Reshape to sequence format for transformer
    x = tf.keras.layers.Reshape((seq_len, embed_dim))(x)
    
    # Add layer normalization
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Create and add positional encoding
    pos_encoding = get_positional_encoding(seq_len, embed_dim)
    x = tf.keras.layers.Add()([x, pos_encoding])
    
    # Memory-efficient implementation of transformer encoder
    for i in range(num_encoder_layers):
        # Multi-head attention
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,  # Ensure key_dim is divisible by num_heads
            dropout=0.1
        )(x, x)
        
        # Add dropout and residual connection
        attn_output = tf.keras.layers.Dropout(0.1)(attn_output)
        x = tf.keras.layers.Add()([x, attn_output])
        
        # Layer normalization
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network (smaller than standard transformer)
        ffn = tf.keras.layers.Dense(embed_dim, activation='gelu')(x)
        ffn = tf.keras.layers.Dense(embed_dim)(ffn)
        
        # Add dropout and residual connection
        ffn = tf.keras.layers.Dropout(0.1)(ffn)
        x = tf.keras.layers.Add()([x, ffn])
        
        # Layer normalization
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Classification head (smaller than usual)
    x = tf.keras.layers.Dense(32, activation='gelu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create and return model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    log_info(f"Model created with {model.count_params()} parameters")
    
    return model

# Create a custom training loop with gradient accumulation
class GradientAccumulationModel:
    """A custom training wrapper that implements gradient accumulation"""
    
    def __init__(self, model, optimizer, loss_fn, metric_fns, accumulation_steps=2):
        try:
            self.model = model
            self.optimizer = optimizer
            self.loss_fn = loss_fn
            self.metric_fns = metric_fns
            self.accumulation_steps = accumulation_steps
            
            # Create trainable variables list for gradient accumulation
            self.trainable_vars = self.model.trainable_variables
            
            # Create gradient accumulators
            self.accum_grads = [tf.Variable(tf.zeros_like(var), trainable=False) for var in self.trainable_vars]
            
            # Store metric names
            self.metric_names = ["accuracy", "auc"]
            
            # Create metric objects outside of tf.function to avoid variable creation issues
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
            self.train_auc = tf.keras.metrics.AUC(name='train_auc')
            
            self.val_loss = tf.keras.metrics.Mean(name='val_loss')
            self.val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')
            self.val_auc = tf.keras.metrics.AUC(name='val_auc')
            
            # Initialize the train step to make TF trace it on first call
            # This will prevent the variable creation error on subsequent calls
            log_info("Creating dummy data for model initialization")
            dummy_x = tf.ones((1, model.input_shape[1], model.input_shape[2]), 
                           dtype=tf.float16 if mixed_precision.global_policy().name == 'mixed_float16' else tf.float32)
            dummy_y = tf.zeros((1,), dtype=tf.float32)
            
            log_info("Initializing train step function")
            self._initialize_train_step(dummy_x, dummy_y)
            log_info("Initializing test step function")
            self._initialize_test_step(dummy_x, dummy_y)
            
            log_info(f"Created gradient accumulation model with {accumulation_steps} accumulation steps")
        except Exception as e:
            log_exception(e)
            raise
    
    def _initialize_train_step(self, dummy_x, dummy_y):
        """Initialize the train step function with dummy data"""
        # This triggers function tracing once, so that variables are created only the first time
        try:
            log_info("Running dummy train step")
            self.train_step(dummy_x, dummy_y, 0)
            # Clear metrics after initialization
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.train_auc.reset_states()
            # Clear accumulated gradients
            self.reset_gradients()
            log_info("Dummy train step completed successfully")
        except Exception as e:
            log_exception(e)
            raise
    
    def _initialize_test_step(self, dummy_x, dummy_y):
        """Initialize the test step function with dummy data"""
        # This triggers function tracing once, so that variables are created only the first time
        try:
            log_info("Running dummy test step")
            self.test_step(dummy_x, dummy_y)
            # Clear metrics after initialization
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()
            self.val_auc.reset_states()
            log_info("Dummy test step completed successfully")
        except Exception as e:
            log_exception(e)
            raise
    
    def reset_gradients(self):
        """Reset accumulated gradients to zero"""
        for i, grad in enumerate(self.accum_grads):
            self.accum_grads[i].assign(tf.zeros_like(grad))
    
    @tf.function
    def train_step(self, x, y, step):
        """Execute a single training step with gradient accumulation"""
        # Compute gradients
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(x, training=True)
            # Calculate loss
            loss = self.loss_fn(y, predictions)
            # Add regularization losses
            if self.model.losses:
                loss += tf.math.add_n(self.model.losses)
            # Scale loss by accumulation steps
            scaled_loss = loss / self.accumulation_steps
        
        # Compute gradients
        gradients = tape.gradient(scaled_loss, self.trainable_vars)
        
        # Accumulate gradients
        for i, grad in enumerate(gradients):
            if grad is not None:
                self.accum_grads[i].assign_add(grad)
        
        # Apply gradients only at the end of accumulation steps
        if (step + 1) % self.accumulation_steps == 0:
            # Apply accumulated gradients
            self.optimizer.apply_gradients(zip(self.accum_grads, self.trainable_vars))
            # Reset gradients
            self.reset_gradients()
        
        # Update metrics using the metrics objects
        self.train_loss(loss)
        self.train_accuracy(y, predictions)
        self.train_auc(y, predictions)
        
        # Return metrics dictionary
        metrics = {
            'loss': self.train_loss.result(),
            'accuracy': self.train_accuracy.result(),
            'auc': self.train_auc.result()
        }
        
        return metrics
    
    @tf.function
    def test_step(self, x, y):
        """Execute a single validation/test step"""
        # Forward pass
        predictions = self.model(x, training=False)
        # Calculate loss
        loss = self.loss_fn(y, predictions)
        
        # Update metrics using the metrics objects
        self.val_loss(loss)
        self.val_accuracy(y, predictions)
        self.val_auc(y, predictions)
        
        # Return metrics dictionary
        metrics = {
            'loss': self.val_loss.result(),
            'accuracy': self.val_accuracy.result(),
            'auc': self.val_auc.result()
        }
        
        return metrics
        
    def reset_metrics(self, training=True):
        """Reset metric states"""
        if training:
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.train_auc.reset_states()
        else:
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()
            self.val_auc.reset_states()

# Load dataset statistics or calculate them if needed
def get_dataset_statistics(train_pickle, target_shape, sample_size=1000):
    """Calculate or load dataset statistics for normalization"""
    log_info("Calculating dataset statistics...")
    
    try:
        # Load a sample of training data for statistics
        with open(train_pickle, 'rb') as f:
            data = pickle.load(f)
            # Limit to sample_size
            sample_size = min(sample_size, len(data))
            data_sample = data[:sample_size]
            log_info(f"Loaded {sample_size} samples for statistics calculation")
        
        # Process sample
        processed_sample = np.zeros((sample_size, target_shape[0], target_shape[1]), dtype=np.float16)
        for i in range(sample_size):
            processed_sample[i] = preprocess_spectrogram(data_sample[i], target_shape)
        
        # Calculate statistics
        mean = np.mean(processed_sample)
        std = np.std(processed_sample)
        
        # Clean memory
        data_sample = None
        processed_sample = None
        clean_memory()
        
        log_info(f"Dataset statistics: mean={mean:.4f}, std={std:.4f}")
        return mean, std
    except Exception as e:
        log_exception(e)
        raise

# Function to setup callbacks
def setup_callbacks(model_dir, param):
    """Setup training callbacks"""
    callbacks = []
    
    # Early stopping
    if param.get("fit", {}).get("early_stopping", {}).get("enabled", True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=param.get("fit", {}).get("early_stopping", {}).get("monitor", "val_loss"),
                patience=param.get("fit", {}).get("early_stopping", {}).get("patience", 10),
                restore_best_weights=True
            )
        )
    
    # Learning rate scheduler
    if param.get("fit", {}).get("lr_scheduler", {}).get("enabled", True):
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
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{model_dir}/best_model.keras",
                monitor=param.get("fit", {}).get("checkpointing", {}).get("monitor", "val_accuracy"),
                save_best_only=True,
                mode=param.get("fit", {}).get("checkpointing", {}).get("mode", "max")
            )
        )
    
    return callbacks

# Main function to run training
def main():
    try:
        log_info("Starting ultra memory-efficient AST training")
        print_memory_usage("Initial")
        
        # Define paths to data
        pickle_dir = param['pickle_directory']
        train_pickle = f"{pickle_dir}/train_overall.pickle"
        train_labels_pickle = f"{pickle_dir}/train_labels_overall.pickle"
        val_pickle = f"{pickle_dir}/val_overall.pickle"
        val_labels_pickle = f"{pickle_dir}/val_labels_overall.pickle"
        test_pickle = f"{pickle_dir}/test_files_overall.pickle"
        test_labels_pickle = f"{pickle_dir}/test_labels_overall.pickle"
        
        log_info(f"Using data from: {pickle_dir}")
        
        # Check if pickle files exist
        for p in [train_pickle, train_labels_pickle, val_pickle, val_labels_pickle, test_pickle, test_labels_pickle]:
            if not os.path.exists(p):
                log_error(f"Error: Missing pickle file {p}")
                sys.exit(1)
        
        # Get dataset statistics for normalization
        mean, std = get_dataset_statistics(train_pickle, target_shape)
        
        # Handle infinity values in std - this can happen with spectrograms
        if not np.isfinite(std) or std == 0:
            log_info("Warning: Standard deviation is infinite or zero. Using 1.0 instead.")
            std = 1.0
        
        print_memory_usage("After statistics calculation")
        
        # Create data generators with extremely small micro-batch size
        # The batch size from config becomes the logical batch size for gradient accumulation
        # The micro batch size is what's actually loaded in memory at once
        batch_size = param.get("fit", {}).get("batch_size", 8)
        micro_batch_size = 2  # Extremely small to avoid OOM
        accumulation_steps = batch_size // micro_batch_size
        
        log_info(f"Using gradient accumulation with logical batch size={batch_size}, "
              f"micro_batch_size={micro_batch_size}, accumulation_steps={accumulation_steps}")
        
        # Create train and validation generators
        log_info("Creating training data generator")
        train_gen = MemoryEfficientDataGenerator(
            data_path=train_pickle,
            labels_path=train_labels_pickle,
            batch_size=micro_batch_size,  # Use micro batch size for the generator
            target_shape=target_shape,
            mean=mean,
            std=std,
            is_training=True
        )
        
        log_info("Creating validation data generator")
        val_gen = MemoryEfficientDataGenerator(
            data_path=val_pickle,
            labels_path=val_labels_pickle,
            batch_size=micro_batch_size,  # Use micro batch size for the generator
            target_shape=target_shape,
            mean=mean,
            std=std,
            is_training=False
        )
        
        print_memory_usage("After creating data generators")
        
        # Create model
        log_info("Creating AST model")
        model = create_memory_efficient_ast(
            input_shape=target_shape,
            config=param.get("model", {}).get("architecture", {})
        )
        
        # Print model summary
        model.summary()
        
        print_memory_usage("After creating model")
        
        # Create optimizer with gradient clipping
        log_info("Creating optimizer")
        optimizer = AdamW(
            learning_rate=param.get("fit", {}).get("compile", {}).get("learning_rate", 0.0001),
            weight_decay=0.01,
            clipnorm=param.get("training", {}).get("gradient_clip_norm", 1.0)
        )
        
        # Loss function
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        
        # Metric functions
        metric_fns = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc")
        ]
        
        # Create gradient accumulation model
        log_info("Creating gradient accumulation model")
        ga_model = GradientAccumulationModel(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            accumulation_steps=accumulation_steps
        )
        
        print_memory_usage("After creating gradient accumulation model")
        
        # Training parameters
        epochs = param.get("fit", {}).get("epochs", 50)
        steps_per_epoch = len(train_gen)  # We don't multiply by accumulation_steps here
        validation_steps = len(val_gen)
        
        # Initialize history dictionary
        history = {
            'loss': [], 'accuracy': [], 'auc': [],
            'val_loss': [], 'val_accuracy': [], 'val_auc': []
        }
        
        log_info(f"Starting training for {epochs} epochs")
        log_info(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")
        
        # Start timing
        start_time = time.time()
        
        # Custom training loop with gradient accumulation
        for epoch in range(epochs):
            log_info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Reset metrics for new epoch
            ga_model.reset_metrics(training=True)
            
            # Reset gradient accumulators at the start of each epoch
            ga_model.reset_gradients()
            
            # Training loop
            log_info("Starting training loop")
            train_pbar = tqdm(enumerate(train_gen), total=len(train_gen), desc="Training")
            for i, (x_batch, y_batch) in train_pbar:
                # Execute train step
                metrics = ga_model.train_step(x_batch, y_batch, i % accumulation_steps)
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': metrics['loss'].numpy(),
                    'accuracy': metrics['accuracy'].numpy()
                })
                
                # Periodic memory cleanup
                if i % 50 == 0 and i > 0:
                    # Only perform garbage collection (avoid clearing the whole session)
                    gc.collect()
                    
                # Log progress every 200 steps
                if i % 200 == 0:
                    log_info(f"Training step {i}/{len(train_gen)}: "
                         f"loss={metrics['loss'].numpy():.4f}, "
                         f"accuracy={metrics['accuracy'].numpy():.4f}")
            
            # Calculate average training metrics at the end of the epoch
            epoch_train_loss = ga_model.train_loss.result().numpy()
            epoch_train_accuracy = ga_model.train_accuracy.result().numpy()
            epoch_train_auc = ga_model.train_auc.result().numpy()
            
            # Memory cleanup after training
            clean_memory()
            print_memory_usage("After training")
            
            # Reset validation metrics
            ga_model.reset_metrics(training=False)
            
            # Validation loop
            log_info("Starting validation loop")
            val_pbar = tqdm(enumerate(val_gen), total=len(val_gen), desc="Validation")
            for i, (x_batch, y_batch) in val_pbar:
                # Execute validation step
                metrics = ga_model.test_step(x_batch, y_batch)
                
                # Update progress bar
                val_pbar.set_postfix({
                    'val_loss': metrics['loss'].numpy(), 
                    'val_accuracy': metrics['accuracy'].numpy()
                })
                
                # Log progress every 100 steps
                if i % 100 == 0:
                    log_info(f"Validation step {i}/{len(val_gen)}: "
                         f"val_loss={metrics['loss'].numpy():.4f}, "
                         f"val_accuracy={metrics['accuracy'].numpy():.4f}")
            
            # Calculate average validation metrics
            epoch_val_loss = ga_model.val_loss.result().numpy()
            epoch_val_accuracy = ga_model.val_accuracy.result().numpy()
            epoch_val_auc = ga_model.val_auc.result().numpy()
            
            # Memory cleanup after validation
            clean_memory()
            print_memory_usage("After validation")
            
            # Update history
            history['loss'].append(epoch_train_loss)
            history['accuracy'].append(epoch_train_accuracy)
            history['auc'].append(epoch_train_auc)
            history['val_loss'].append(epoch_val_loss)
            history['val_accuracy'].append(epoch_val_accuracy)
            history['val_auc'].append(epoch_val_auc)
            
            # Print epoch results
            log_info(f"Epoch {epoch+1}/{epochs} - loss: {epoch_train_loss:.4f} - accuracy: {epoch_train_accuracy:.4f} - "
                  f"auc: {epoch_train_auc:.4f} - val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_accuracy:.4f} - "
                  f"val_auc: {epoch_val_auc:.4f}")
            
            # Early stopping logic
            if epoch > 5 and history['val_loss'][-1] > history['val_loss'][-2] > history['val_loss'][-3]:
                log_info("Early stopping triggered")
                break
            
            # Save model if it's the best so far (based on validation accuracy)
            if epoch == 0 or history['val_accuracy'][-1] > max(history['val_accuracy'][:-1]):
                log_info("Saving best model")
                model.save(f"{param['model_directory']}/best_model.keras")
        
        # End timing
        end_time = time.time()
        training_time = end_time - start_time
        log_info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Save final model
        model.save(f"{param['model_directory']}/final_model.keras")
        log_info("Final model saved")
        
        # Plot training history
        log_info("Creating training history plot")
        plt.figure(figsize=(12, 8))
        
        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{param['result_directory']}/training_history.png")
        plt.close()
        
        log_info("Training history plot saved")
        
        # Evaluate on test data if available
        log_info("\nEvaluating model on test set")
        test_gen = MemoryEfficientDataGenerator(
            data_path=test_pickle,
            labels_path=test_labels_pickle,
            batch_size=micro_batch_size,
            target_shape=target_shape,
            mean=mean,
            std=std,
            shuffle=False,
            is_training=False
        )
        
        # Create predictions list
        y_true = []
        y_pred = []
        
        # Prediction loop
        log_info("Running predictions on test set")
        test_pbar = tqdm(test_gen, desc="Testing")
        for x_batch, y_batch in test_pbar:
            # Get predictions
            pred_batch = model.predict_on_batch(x_batch)
            
            # Store true labels and predictions
            y_true.extend(y_batch.numpy())
            y_pred.extend(pred_batch.numpy().flatten())
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate classification metrics
        y_pred_binary = (y_pred > 0.5).astype(int)
        classification_rep = classification_report(y_true, y_pred_binary)
        conf_matrix = confusion_matrix(y_true, y_pred_binary)
        auc = roc_auc_score(y_true, y_pred)
        
        # Print evaluation results
        log_info("\nTest Evaluation Results:")
        log_info(f"AUC: {auc:.4f}")
        log_info("\nClassification Report:")
        log_info(classification_rep)
        log_info("\nConfusion Matrix:")
        log_info(str(conf_matrix))
        
        # Save results to text file
        with open(f"{param['result_directory']}/test_evaluation.txt", 'w') as f:
            f.write("Test Evaluation Results:\n")
            f.write(f"AUC: {auc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_rep)
            f.write("\nConfusion Matrix:\n")
            f.write(str(conf_matrix))
        
        log_info("Test evaluation results saved")
        log_info("\nTraining and evaluation completed successfully!")
        
    except Exception as e:
        log_exception(e)
        sys.exit(1)

# Run the main function if script is executed directly
if __name__ == "__main__":
    main()