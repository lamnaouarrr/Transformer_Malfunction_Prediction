#!/usr/bin/env python3
"""
memory_optimized_trainer.py - A memory-efficient trainer for AST model
This script loads and processes data in batches to prevent GPU memory issues
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
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import gc  # For garbage collection

# Make sure directories exist
os.makedirs("./model/AST", exist_ok=True)
os.makedirs("./result/result_AST", exist_ok=True)

# Configure GPU memory growth
print("Configuring GPU memory growth...")
for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Set memory growth for {gpu}")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# Limit GPU memory usage
print("Limiting GPU memory usage...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Limit TensorFlow to only allocate 400MB of memory on the first GPU
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=400)]
        )
        print("Limited GPU memory to 400MB")
    except RuntimeError as e:
        print(f"Error limiting GPU memory: {e}")

# Enable XLA for faster training
print("Enabling XLA acceleration...")
tf.config.optimizer.set_jit(True)

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
mixed_precision.set_global_policy('mixed_float16')
print(f"Mixed precision policy: {mixed_precision.global_policy()}")

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
        print(f"Found: {p}")
    else:
        print(f"Missing: {p}")
        sys.exit(1)

# Get file sizes
print("Checking file sizes:")
for p in [train_pickle, train_labels_pickle, val_pickle, val_labels_pickle, test_files_pickle, test_labels_pickle]:
    size_mb = os.path.getsize(p) / (1024 * 1024)
    print(f"{p}: {size_mb:.2f} MB")

# Function to load pickle files
def load_pickle(file_path):
    print(f"Loading: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Function to preprocess a single spectrogram to ensure consistent shape
def preprocess_spectrogram(spec, target_shape):
    # Handle 3D input (when a single frame has an extra dimension)
    if len(spec.shape) == 3 and spec.shape[2] == 1:
        spec = spec[:, :, 0]  # Remove the last dimension
    
    # Skip if dimensions already match
    if spec.shape[0] == target_shape[0] and spec.shape[1] == target_shape[1]:
        return spec
        
    try:
        # Simple resize using zero padding/cropping
        temp_spec = np.zeros(target_shape, dtype=np.float32)
        freq_dim = min(spec.shape[0], target_shape[0])
        time_dim = min(spec.shape[1], target_shape[1])
        temp_spec[:freq_dim, :time_dim] = spec[:freq_dim, :time_dim]
        return temp_spec
    except Exception as e:
        print(f"Error processing spectrogram: {e}")
        # Fill with zeros if processing fails
        return np.zeros(target_shape, dtype=np.float32)

# Define target shape for spectrograms
target_shape = (param["feature"]["n_mels"], 96)
print(f"Target spectrogram shape: {target_shape}")

# Load only validation data and labels completely (it's smaller)
print("\nLoading validation data...")
val_data = load_pickle(val_pickle)
val_labels = load_pickle(val_labels_pickle)
print(f"Validation data shape: {val_data.shape}")
print(f"Validation labels shape: {val_labels.shape}")

# Preprocess validation data
print("Preprocessing validation data...")
processed_val_data = np.zeros((val_data.shape[0], target_shape[0], target_shape[1]), dtype=np.float32)
for i in range(val_data.shape[0]):
    processed_val_data[i] = preprocess_spectrogram(val_data[i], target_shape)
    if i % 500 == 0 and i > 0:
        print(f"Processed {i}/{val_data.shape[0]} validation samples")

# Free memory
val_data = None
gc.collect()

print(f"Processed validation data shape: {processed_val_data.shape}")

# Calculate normalization statistics on a sample of training data
print("\nLoading a sample of training data for normalization statistics...")
with open(train_pickle, 'rb') as f:
    # Read first 1000 samples for statistics
    train_data_sample = pickle.load(f)[:1000]

# Preprocess sample for statistics
processed_sample = np.zeros((train_data_sample.shape[0], target_shape[0], target_shape[1]), dtype=np.float32)
for i in range(train_data_sample.shape[0]):
    processed_sample[i] = preprocess_spectrogram(train_data_sample[i], target_shape)

# Calculate mean and std from sample
mean = np.mean(processed_sample)
std = np.std(processed_sample)
print(f"Training data statistics (from sample) - Mean: {mean:.4f}, Std: {std:.4f}")

# Free memory
train_data_sample = None
processed_sample = None
gc.collect()

# Normalize validation data
if std > 0:
    processed_val_data = (processed_val_data - mean) / std
    print("Z-score normalization applied to validation data")
else:
    print("Warning: Standard deviation is 0, skipping normalization")

# Load validation labels
val_labels = np.array(val_labels, dtype=np.float32)

# Create validation dataset
print("Creating validation dataset...")
val_dataset = tf.data.Dataset.from_tensor_slices((processed_val_data, val_labels))
val_dataset = val_dataset.batch(param.get("fit", {}).get("batch_size", 16)).prefetch(tf.data.AUTOTUNE)

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

# Function to create the AST model (simplified version for memory efficiency)
def create_ast_model(input_shape, config=None):
    print(f"Creating memory-efficient AST model with input shape {input_shape}...")
    if config is None:
        config = {}
    
    # Get transformer configuration
    transformer_config = config.get("transformer", {})
    num_heads = transformer_config.get("num_heads", 1)  # Reduced from 4
    dim_feedforward = transformer_config.get("dim_feedforward", 64)  # Reduced from 128
    num_encoder_layers = transformer_config.get("num_encoder_layers", 1)  # Keep at 1
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
        ffn = tf.keras.layers.Dense(embed_dim * 2, activation='gelu')(x)  # Reduced multiplier from 4 to 2
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

# Create a custom data generator that loads and processes batches on the fly
class TrainingDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path, labels_path, batch_size, target_shape, mean, std):
        # Load all labels (they're small enough)
        self.labels = load_pickle(labels_path)
        self.n_samples = len(self.labels)
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.mean = mean
        self.std = std
        
        # Load the data file handle but don't read all data at once
        self.data_file = open(data_path, 'rb')
        self.data = pickle.load(self.data_file)
        self.data_file.close()
        
        # Create shuffled indices
        self.indices = np.random.permutation(self.n_samples)
        
        print(f"Created generator with {self.n_samples} samples")
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, idx):
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Open the data file for each batch
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
            
            # Initialize batch arrays
            batch_x = np.zeros((len(batch_indices), self.target_shape[0], self.target_shape[1]), dtype=np.float32)
            batch_y = np.zeros((len(batch_indices),), dtype=np.float32)
            
            # Fill the arrays
            for i, idx in enumerate(batch_indices):
                if idx < len(data):
                    # Preprocess the spectrogram
                    batch_x[i] = preprocess_spectrogram(data[idx], self.target_shape)
                    batch_y[i] = self.labels[idx]
            
            # Apply normalization
            if self.std > 0:
                batch_x = (batch_x - self.mean) / self.std
            
        return batch_x, batch_y
    
    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch
        self.indices = np.random.permutation(self.n_samples)

# Create model
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

# Create training data generator
print("Creating training data generator...")
batch_size = param.get("fit", {}).get("batch_size", 16)
train_generator = TrainingDataGenerator(
    data_path=train_pickle,
    labels_path=train_labels_pickle,
    batch_size=batch_size,
    target_shape=target_shape,
    mean=mean,
    std=std
)

# Train model
print(f"Starting model training for {param['fit']['epochs']} epochs...")
training_start_time = time.time()

history = model.fit(
    train_generator,
    epochs=param["fit"]["epochs"],
    validation_data=val_dataset,
    verbose=1,
    callbacks=callbacks,
    workers=1,
    use_multiprocessing=False  # Set to False to avoid memory issues
)

training_end_time = time.time()
training_duration = training_end_time - training_start_time
print(f"Training completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")

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

print("\nMemory-optimized training script completed successfully!")