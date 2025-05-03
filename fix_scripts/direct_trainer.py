#!/usr/bin/env python3
"""
direct_trainer.py - A standalone script to directly load pickle files and train the AST model
This bypasses all the complex code in baseline_AST.py that's causing issues
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

# Make sure directories exist
os.makedirs("./model/AST", exist_ok=True)
os.makedirs("./result/result_AST", exist_ok=True)

# Configure GPU memory growth (prevents TF from allocating all GPU memory at once)
print("Configuring GPU memory growth...")
for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Set memory growth for {gpu}")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# Enable XLA for faster training
print("Enabling XLA acceleration...")
tf.config.optimizer.set_jit(True)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

# Load configuration
print("Loading configuration...")
try:
    with open("baseline_AST.yaml", "r") as stream:
        param = yaml.safe_load(stream)
        print("Configuration loaded successfully")
except Exception as e:
    print(f"Error loading configuration: {e}")
    sys.exit(1)

# Enable mixed precision for faster training (if supported by GPU)
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

# Function to load pickle files
def load_pickle(file_path):
    print(f"Loading: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Load pickle files
print("\nLoading pickle files...")
try:
    train_data = load_pickle(train_pickle)
    train_labels = load_pickle(train_labels_pickle)
    val_data = load_pickle(val_pickle)
    val_labels = load_pickle(val_labels_pickle)
    test_files = load_pickle(test_files_pickle)
    test_labels = load_pickle(test_labels_pickle)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Validation labels shape: {val_labels.shape}")
    print(f"Test files count: {len(test_files)}")
    print(f"Test labels shape: {test_labels.shape}")
except Exception as e:
    print(f"Error loading pickle files: {e}")
    sys.exit(1)

# Ensure all data is in the expected format
if not isinstance(train_data, np.ndarray) or train_data.shape[0] == 0:
    print("Error: Train data is not a valid numpy array or is empty")
    sys.exit(1)
if not isinstance(val_data, np.ndarray) or val_data.shape[0] == 0:
    print("Error: Validation data is not a valid numpy array or is empty")
    sys.exit(1)

# Define target shape for spectrograms
target_shape = (param["feature"]["n_mels"], 96)
print(f"Target spectrogram shape: {target_shape}")

# Function to preprocess spectrograms to ensure consistent shape
def preprocess_spectrograms(spectrograms, target_shape):
    print(f"Preprocessing spectrograms to shape {target_shape}...")
    if spectrograms.shape[0] == 0:
        return spectrograms
    
    batch_size = spectrograms.shape[0]
    processed = np.zeros((batch_size, target_shape[0], target_shape[1]), dtype=np.float32)
    
    for i in range(batch_size):
        spec = spectrograms[i]
        
        # Handle 3D input (when a single frame has an extra dimension)
        if len(spec.shape) == 3 and spec.shape[2] == 1:
            spec = spec[:, :, 0]  # Remove the last dimension
        
        # Skip if dimensions already match
        if spec.shape[0] == target_shape[0] and spec.shape[1] == target_shape[1]:
            processed[i] = spec
            continue
            
        try:
            # Simple resize using zero padding/cropping
            temp_spec = np.zeros(target_shape, dtype=np.float32)
            freq_dim = min(spec.shape[0], target_shape[0])
            time_dim = min(spec.shape[1], target_shape[1])
            temp_spec[:freq_dim, :time_dim] = spec[:freq_dim, :time_dim]
            processed[i] = temp_spec
        except Exception as e:
            print(f"Error processing spectrogram at index {i}: {e}")
            # Fill with zeros if processing fails
            processed[i] = np.zeros(target_shape, dtype=np.float32)
    
    return processed

# Preprocess data to ensure consistent shapes
train_data = preprocess_spectrograms(train_data, target_shape)
val_data = preprocess_spectrograms(val_data, target_shape)

print(f"Preprocessed train data shape: {train_data.shape}")
print(f"Preprocessed validation data shape: {val_data.shape}")

# Normalize data
print("Normalizing data...")
mean = np.mean(train_data)
std = np.std(train_data)
print(f"Training data statistics - Mean: {mean:.4f}, Std: {std:.4f}")

if std > 0:
    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std
    print("Z-score normalization applied")
else:
    print("Warning: Standard deviation is 0, skipping normalization")

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
    print(f"Creating AST model with input shape {input_shape}...")
    if config is None:
        config = {}
    
    # Get transformer configuration
    transformer_config = config.get("transformer", {})
    num_heads = transformer_config.get("num_heads", 4)
    dim_feedforward = transformer_config.get("dim_feedforward", 512)
    num_encoder_layers = transformer_config.get("num_encoder_layers", 3)
    patch_size = transformer_config.get("patch_size", 4)
    attention_dropout = transformer_config.get("attention_dropout", 0.2)
    
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

# Create TensorFlow datasets
print("Creating TensorFlow datasets...")
batch_size = param.get("fit", {}).get("batch_size", 16)
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_dataset = train_dataset.batch(batch_size).shuffle(1000).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Train model
print(f"Starting model training for {param['fit']['epochs']} epochs...")
training_start_time = time.time()

history = model.fit(
    train_dataset,
    epochs=param["fit"]["epochs"],
    validation_data=val_dataset,
    verbose=1,
    callbacks=callbacks
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

print("\nDirect training script completed successfully!")