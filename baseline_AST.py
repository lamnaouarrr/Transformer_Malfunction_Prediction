#!/usr/bin/env python
"""
 @file   baseline_AST.py
 @brief  Baseline code of simple AE-based anomaly detection used experiment in [1], updated for 2025 with enhancements.
 @author Ryo Tanabe and Yohei Kawaguchi (Hitachi Ltd.), updated by Lamnaouar Ayoub (Github: lamnaouarrr)
 Copyright (C) 2019 Hitachi, Ltd. All right reserved.
 [1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," arXiv preprint arXiv:1909.09347, 2019.
"""
########################################################################
# import default python-library
########################################################################
import pickle
import os
import sys
import glob
import logging
import numpy as np
import librosa
import librosa.core
import librosa.feature
import yaml
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import seaborn as sns
import math
import gc
import warnings

# Configure memory optimizations (integrated from run_optimized_trainer.py)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
warnings.filterwarnings('ignore')

# Memory optimization environment variables
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MEMORY_ALLOCATION'] = '0.8'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2'

from pathlib import Path
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, Add, MultiHeadAttention, LayerNormalization, Reshape, Permute, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.losses import mse as mean_squared_error
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from skimage.metrics import structural_similarity as ssim
try:
    from transformers import TFViTModel, ViTConfig
except ImportError:
    print("Warning: transformers package not installed. Some features may not work.")
    print("Install with: pip install transformers")

# Add prominent startup message
print("\n" + "="*80)
print(" AUDIO SPECTROGRAM TRANSFORMER (AST) FOR MACHINE MALFUNCTION DETECTION ")
print("="*80 + "\n")

def print_with_timestamp(message):
    """Print a message with a timestamp for better logging"""
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{current_time}] {message}")

########################################################################

########################################################################
# version
########################################################################
__versions__ = "3.0.0"
########################################################################

_SPECTROGRAM_CACHE = {}

def binary_cross_entropy_loss(y_true, y_pred):
    """
    Binary cross-entropy loss for autoencoder with improved memory efficiency
    """
    # Use TF's built-in binary_crossentropy for better memory efficiency
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal loss for addressing class imbalance with improved numerical stability
    """
    # Convert inputs to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Clip predictions to prevent numerical instability
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # Binary cross entropy
    bce = K.binary_crossentropy(y_true, y_pred)
    
    # Focal weight
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = K.pow(1.0 - p_t, gamma)
    
    # Apply weights
    focal_loss = alpha_factor * modulating_factor * bce
    
    return K.mean(focal_loss)

def positional_encoding(seq_len, d_model, encoding_type="sinusoidal"):
    """
    Create positional encodings for the transformer model
    
    Args:
        seq_len: sequence length (int)
        d_model: depth of the model (int)
        encoding_type: type of positional encoding
        
    Returns:
        Positional encoding tensor with shape (1, seq_len, d_model)
    """
    # Create position vector
    positions = np.arange(seq_len)[:, np.newaxis]
    
    # Create dimension vector
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Create encoding
    pe = np.zeros((seq_len, d_model))
    
    # Apply sin to even indices
    pe[:, 0::2] = np.sin(positions * div_term)
    
    # Apply cos to odd indices
    pe[:, 1::2] = np.cos(positions * div_term)
    
    # Add batch dimension and convert to tensor
    pe = np.expand_dims(pe, axis=0)
    
    return tf.cast(pe, dtype=tf.float32)

########################################################################
# setup STD I/O
########################################################################
def setup_logging():
    os.makedirs("./logs/log__AST", exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, filename="./logs/log_AST/baseline__AST.log")
    logger = logging.getLogger(' ')
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logging()
########################################################################

########################################################################
# visualizer
########################################################################
class Visualizer:
    def __init__(self, param=None):
        self.param = param or {}  # Initialize with param or empty dict

    def loss_plot(self, history, machine_type=None, machine_id=None, db=None):
        # Change this line
        fig_size = self.param.get("visualization", {}).get("figure_size", [30, 20])
        plt.figure(figsize=(fig_size[0], fig_size[1]))
        
        # Create title with machine information
        title_info = ""
        
        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f"Model loss{title_info}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Train", "Validation"], loc="upper right")
        
        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f"Model accuracy{title_info}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["Train", "Validation"], loc="lower right")

    def save_figure(self, name):
        plt.savefig(name)
        plt.close()


    def plot_confusion_matrix(self, y_true, y_pred, classes=None, title=None):
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            if classes is None:
                classes = ['Normal', 'Abnormal']
                
            if title is None:
                title = 'Confusion Matrix'
                
            # Create figure
            fig_size = self.param.get("visualization", {}).get("figure_size", [10, 8])
            plt.figure(figsize=(fig_size[0]//2, fig_size[1]//2))
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=classes, yticklabels=classes)
            plt.title(title)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

########################################################################
# file I/O
########################################################################
def save_pickle(filename, save_data):
    logger.info(f"save_pickle -> {filename}")
    with open(filename, 'wb') as sf:
        pickle.dump(save_data, sf)

def load_pickle(filename):
    logger.info(f"load_pickle <- {filename}")
    with open(filename, 'rb') as lf:
        load_data = pickle.load(lf)
    return load_data

def file_load(wav_name, mono=False):
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except Exception as e:
        logger.error(f"file_broken or not exists!! : {wav_name}, error: {e}")
        return None

def demux_wav(wav_name, channel=0):
    try:
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data is None:
            return None, None
        if multi_channel_data.ndim <= 1:
            return sr, multi_channel_data
        return sr, np.array(multi_channel_data)[channel, :]
    except ValueError as msg:
        logger.warning(f'{msg}')
        return None, None

########################################################################
# feature extractor
########################################################################
def augment_spectrogram(spectrogram, param=None):
    """
    Apply subtle augmentations to spectrograms: frequency/time masking and noise injection.
    """
    if param is None:
        param = {}
    
    aug_config = param.get("feature", {}).get("augmentation", {})
    max_mask_freq = aug_config.get("max_mask_freq", 10)
    max_mask_time = aug_config.get("max_mask_time", 10)
    n_freq_masks = aug_config.get("n_freq_masks", 2)
    n_time_masks = aug_config.get("n_time_masks", 2)
    noise_level = aug_config.get("noise_level", 0.01)

    # Frequency masking
    freq_mask_param = np.random.randint(0, max_mask_freq)
    for _ in range(n_freq_masks):
        freq_start = np.random.randint(0, spectrogram.shape[0])
        freq_end = min(spectrogram.shape[0], freq_start + freq_mask_param)
        spectrogram[freq_start:freq_end, :] *= 0.5  # Reduce amplitude instead of zeroing

    # Time masking
    time_mask_param = np.random.randint(0, max_mask_time)
    for _ in range(n_time_masks):
        time_start = np.random.randint(0, spectrogram.shape[1])
        time_end = min(spectrogram.shape[1], time_start + time_mask_param)
        spectrogram[:, time_start:time_end] *= 0.5

    # Add subtle noise
    noise = np.random.normal(0, noise_level, spectrogram.shape)
    spectrogram += noise

    return spectrogram

def augment_audio(y, sr, param=None):
    """
    Apply advanced audio augmentations directly to the waveform
    
    Args:
        y: Audio waveform
        sr: Sample rate
        param: Parameters dictionary
        
    Returns:
        Augmented audio waveform
    """
    if param is None or not param.get("feature", {}).get("audio_augmentation", {}).get("enabled", False):
        return y
    
    aug_config = param.get("feature", {}).get("audio_augmentation", {})
    
    # Make a copy of the input audio
    y_aug = np.copy(y)
    
    # Apply random time stretching
    if aug_config.get("time_stretch", {}).get("enabled", False) and np.random.rand() < aug_config.get("time_stretch", {}).get("probability", 0.5):
        stretch_factor = np.random.uniform(
            aug_config.get("time_stretch", {}).get("min_factor", 0.8),
            aug_config.get("time_stretch", {}).get("max_factor", 1.2)
        )
        y_aug = librosa.effects.time_stretch(y_aug, rate=stretch_factor)
    
    # Apply random pitch shifting
    if aug_config.get("pitch_shift", {}).get("enabled", False) and np.random.rand() < aug_config.get("pitch_shift", {}).get("probability", 0.5):
        n_steps = np.random.uniform(
            aug_config.get("pitch_shift", {}).get("min_steps", -3),
            aug_config.get("pitch_shift", {}).get("max_steps", 3)
        )
        y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=n_steps)
    
    # Add background noise
    if aug_config.get("background_noise", {}).get("enabled", False) and np.random.rand() < aug_config.get("background_noise", {}).get("probability", 0.5):
        noise_factor = np.random.uniform(
            aug_config.get("background_noise", {}).get("min_factor", 0.001),
            aug_config.get("background_noise", {}).get("max_factor", 0.02)
        )
        noise = np.random.randn(len(y_aug))
        y_aug = y_aug + noise_factor * noise
    
    return y_aug

def file_to_spectrogram(file_name,
                        n_mels=64,
                        n_fft=1024,
                        hop_length=512,
                        power=2.0,
                        augment=False,
                        param=None):
    """
    Convert file_name to a 2D spectrogram with optional augmentation for normal data.
    """
    if isinstance(file_name, (tf.Tensor, np.ndarray)) and len(getattr(file_name, 'shape', [])) >= 2:
        # Already a spectrogram, return it directly
        return file_name
    # Check if file_name is valid
    if not isinstance(file_name, str) or len(file_name) > 1000:
        logger.warning(f"Invalid file name: {type(file_name)}, returning None")
        return None
    
    # Check cache first (skip cache if augmentation is on)
    cache_key = f"{file_name}_{n_mels}_{n_fft}_{hop_length}_{power}"
    if not augment and cache_key in _SPECTROGRAM_CACHE:
        return _SPECTROGRAM_CACHE[cache_key]

    try:
        sr, y = demux_wav(file_name)
        # Use sample rate from parameters if provided
        if param and "feature" in param and "sr" in param["feature"]:
            y = librosa.resample(y, orig_sr=sr, target_sr=param["feature"]["sr"])
            sr = param["feature"]["sr"]
            
        # Apply audio augmentations if enabled
        if augment and y is not None:
            y = augment_audio(y, sr, param)

        if y is None:
            logger.error(f"Failed to load {file_name}")
            return None
        
        # Skip files that are too short
        if len(y) < n_fft:
            logger.warning(f"File too short: {file_name}")
            return None
        
        # For V100, we can use more efficient computation
        # Use more efficient computation for mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                        sr=sr,
                                                        n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        n_mels=n_mels,
                                                        power=power)
        
        # Convert to decibel scale
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        if augment and param is not None and param.get("feature", {}).get("augmentation", {}).get("enabled", False):
            log_mel_spectrogram = augment_spectrogram(log_mel_spectrogram, param)
        
        # Process frames and stride if specified in parameters
        if param and "feature" in param:
            frames = param["feature"].get("frames", None)
            stride = param["feature"].get("stride", None)
            
            # Check if we're in progressive training mode
            in_progressive_mode = param.get("feature", {}).get("target_shape", None) is not None
            
            # If in progressive mode, don't use frames/stride processing
            if frames and stride and not in_progressive_mode:
                # Create frame-based spectrograms
                frame_length = log_mel_spectrogram.shape[1]
                frame_samples = []
                
                # Extract frames with stride
                for i in range(0, max(1, frame_length - frames + 1), stride):
                    if i + frames <= frame_length:
                        frame = log_mel_spectrogram[:, i:i+frames]
                        frame_samples.append(frame)
                
                if frame_samples:
                    # Stack frames or return the sequence
                    return np.stack(frame_samples)
                else:
                    # Pad if needed and return a single frame
                    if log_mel_spectrogram.shape[1] < frames:
                        pad_width = frames - log_mel_spectrogram.shape[1]
                        log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
                    return np.expand_dims(log_mel_spectrogram[:, :frames], axis=0)
        
        # Normalize for model input
        if np.max(log_mel_spectrogram) > np.min(log_mel_spectrogram):
            log_mel_spectrogram = (log_mel_spectrogram - np.min(log_mel_spectrogram)) / (np.max(log_mel_spectrogram) - np.min(log_mel_spectrogram))
        
        # If we have a target shape, resize to it
        target_shape = param.get("feature", {}).get("target_shape", None)
        if target_shape:
            # Ensure we have a 2D spectrogram of the right shape
            if log_mel_spectrogram.shape[0] != target_shape[0] or log_mel_spectrogram.shape[1] != target_shape[1]:
                try:
                    from skimage.transform import resize
                    log_mel_spectrogram = resize(log_mel_spectrogram, target_shape, anti_aliasing=True, mode='reflect')
                except Exception as e:
                    logger.error(f"Error resizing spectrogram: {e}")
                    # Fall back to simple padding/cropping
                    temp_spec = np.zeros(target_shape, dtype=np.float32)
                    freq_dim = min(log_mel_spectrogram.shape[0], target_shape[0])
                    time_dim = min(log_mel_spectrogram.shape[1], target_shape[1])
                    temp_spec[:freq_dim, :time_dim] = log_mel_spectrogram[:freq_dim, :time_dim]
                    log_mel_spectrogram = temp_spec
        else:
            # Ensure minimum size for transformer input - increased for V100
            min_time_dim = param.get("model", {}).get("architecture", {}).get("transformer", {}).get("patch_size", 16) * 4
            if log_mel_spectrogram.shape[1] < min_time_dim:
                # Pad if too short
                pad_width = min_time_dim - log_mel_spectrogram.shape[1]
                log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
            
            # Trim or sample if too long - increased for V100
            max_time_dim = 1024  # Increased from 512 for V100
            if log_mel_spectrogram.shape[1] > max_time_dim:
                start = np.random.randint(0, log_mel_spectrogram.shape[1] - max_time_dim)
                log_mel_spectrogram = log_mel_spectrogram[:, start:start+max_time_dim]

        if not augment and len(_SPECTROGRAM_CACHE) < 10000:  # Limit cache size
            _SPECTROGRAM_CACHE[cache_key] = log_mel_spectrogram
            
        return log_mel_spectrogram
        
    except Exception as e:
        logger.error(f"Error in file_to_spectrogram for {file_name}: {e}")
        return None

def list_to_spectrograms(file_list, labels=None, msg="calc...", augment=False, param=None, batch_size=64):
    """
    Process a list of files into spectrograms with optional labels - memory optimized version
    """
    n_mels = param.get("feature", {}).get("n_mels", 64)
    n_fft = param.get("feature", {}).get("n_fft", 1024)
    hop_length = param.get("feature", {}).get("hop_length", 512)
    power = param.get("feature", {}).get("power", 2.0)
    
    # Handle edge case of empty file list - return empty arrays with correct shapes
    if not file_list or len(file_list) == 0:
        # Use target shape from parameters if available
        target_shape = param.get("feature", {}).get("target_shape", [64, 96])
        if isinstance(target_shape, (list, tuple)) and len(target_shape) == 2:
            shape = target_shape
        else:
            shape = (n_mels, 64)  # Default shape
            
        logger.info(f"Empty file list provided to list_to_spectrograms, returning empty array with shape ({0}, {shape[0]}, {shape[1]})")
        empty_specs = np.zeros((0, shape[0], shape[1]), dtype=np.float32)
        if labels is not None:
            empty_labels = np.array([], dtype=np.float32)
            return empty_specs, empty_labels
        return empty_specs
    
    # First pass: determine dimensions and count valid files
    valid_files = []
    valid_labels = [] if labels is not None else None
    max_freq = 0
    max_time = 0
    
    logger.info(f"First pass: checking dimensions of {len(file_list)} files")
    for idx, file_path in enumerate(tqdm(file_list, desc=f"{msg} (dimension check)")):
        try:
            spec = file_to_spectrogram(file_path, n_mels, n_fft, hop_length, power, augment, param)
            if spec is not None:
                # Handle 3D input
                if len(spec.shape) == 3:
                    spec = spec[0]
                
                max_freq = max(max_freq, spec.shape[0])
                max_time = max(max_time, spec.shape[1])
                valid_files.append(file_path)
                if labels is not None:
                    valid_labels.append(labels[idx])
        except Exception as e:
            logger.error(f"Error checking dimensions for {file_path}: {e}")
    
    # Use target shape from parameters if available
    target_shape = param.get("feature", {}).get("target_shape", None)
    if target_shape:
        max_freq, max_time = target_shape
    else:
        # Round up to nearest multiple of 8 for better GPU utilization
        max_freq = ((max_freq + 7) // 8) * 8
        max_time = ((max_time + 7) // 8) * 8
    
    logger.info(f"Using target shape: ({max_freq}, {max_time})")
    
    # Check if we have any valid files
    if not valid_files:
        logger.warning("No valid files found after dimension check")
        empty_specs = np.zeros((0, max_freq, max_time), dtype=np.float32)
        if labels is not None:
            empty_labels = np.array([], dtype=np.float32)
            return empty_specs, empty_labels
        return empty_specs
    
    # Second pass: process files in batches
    total_valid = len(valid_files)
    spectrograms = np.zeros((total_valid, max_freq, max_time), dtype=np.float32)
    processed_labels = np.array(valid_labels) if valid_labels else None
    
    for batch_start in tqdm(range(0, total_valid, batch_size), desc=f"{msg} (processing)"):
        batch_end = min(batch_start + batch_size, total_valid)
        batch_files = valid_files[batch_start:batch_end]
        
        for i, file_path in enumerate(batch_files):
            try:
                spec = file_to_spectrogram(file_path, n_mels, n_fft, hop_length, power, augment, param)
                
                if spec is not None:
                    # Handle 3D input
                    if len(spec.shape) == 3:
                        spec = spec[0]
                    
                    # Resize if needed
                    if spec.shape[0] != max_freq or spec.shape[1] != max_time:
                        try:
                            from skimage.transform import resize
                            spec = resize(spec, (max_freq, max_time), anti_aliasing=True, mode='reflect')
                        except Exception as e:
                            # Fall back to simple padding/cropping
                            temp_spec = np.zeros((max_freq, max_time))
                            freq_dim = min(spec.shape[0], max_freq)
                            time_dim = min(spec.shape[1], max_time)
                            temp_spec[:freq_dim, :time_dim] = spec[:freq_dim, :time_dim]
                            spec = temp_spec
                    
                    # Store in output array
                    spectrograms[batch_start + i] = spec
                    
                    # Clear memory
                    del spec
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                # Fill with zeros for failed files
                spectrograms[batch_start + i] = np.zeros((max_freq, max_time))
        
        gc.collect()

    spectrograms = spectrograms.astype(np.float32)
    
    if labels is not None:
        return spectrograms, processed_labels
    return spectrograms

def dataset_generator(target_dir, param=None):
    """
    Generate training, validation, and testing datasets for the new simplified directory structure.
    
    target_dir: Base directory ('normal' or 'abnormal')
    param: parameters dictionary from the YAML config
    """
    print_with_timestamp(f"DEBUG: dataset_generator called with target_dir: {target_dir}")
    logger.info(f"target_dir : {target_dir}")
    
    if param is None:
        param = {}
    
    split_ratio = param.get("dataset", {}).get("split_ratio", [0.8, 0.1, 0.1])
    ext = param.get("dataset", {}).get("file_extension", "wav")
    
    # Determine if we're processing normal or abnormal files
    is_normal = "normal" in str(target_dir)
    condition = "normal" if is_normal else "abnormal"
    print_with_timestamp(f"DEBUG: Processing {condition} files")
    
    # Get all files in the directory
    files_in_dir = list(Path(target_dir).glob(f"*.{ext}"))
    print_with_timestamp(f"Looking for files in: {target_dir}")
    print_with_timestamp(f"Found {len(files_in_dir)} files")

    # If no files found, try listing the directory contents
    if len(files_in_dir) == 0:
        print_with_timestamp(f"DEBUG: No files with extension '{ext}' found in {target_dir}")
        print_with_timestamp(f"DEBUG: Directory contents: {list(Path(target_dir).iterdir())[:10]}")  # Show first 10 files
    
    # Parse file names to extract db, machine_type, and machine_id
    normal_data = {}  # {(db, machine_type, machine_id): [files]}
    abnormal_data = {}  # {(db, machine_type, machine_id): [files]}
    
    # Process current directory files
    for file_path in files_in_dir:
        filename = file_path.name
        parts = filename.split('_')
        
        if len(parts) >= 4:  # Ensure we have enough parts in the filename
            # Format: normal_0dB_fan_id_00-00000000.wav
            # Or: abnormal_0dB_fan_id_00-00000000.wav
            condition = parts[0]  # normal or abnormal
            db = parts[1]
            machine_type = parts[2]
            # Handle id part which might contain a hyphen and file number
            id_part = parts[3]
            machine_id = id_part.split('-')[0] if '-' in id_part else id_part
            
            key = (db, machine_type, machine_id)
            
            # Add file to appropriate dictionary based on its actual condition
            if condition == "normal":
                if key not in normal_data:
                    normal_data[key] = []
                normal_data[key].append(str(file_path))
            else:
                if key not in abnormal_data:
                    abnormal_data[key] = []
                abnormal_data[key].append(str(file_path))
    
    # Get the other condition directory to find matching files
    other_condition = "abnormal" if is_normal else "normal"
    other_dir = Path(param.get("base_directory", "./dataset")) / other_condition
    
    # Get files from the other condition directory
    other_files = list(other_dir.glob(f"*.{ext}"))
    
    # Process other directory files
    for file_path in other_files:
        filename = file_path.name
        parts = filename.split('_')
        
        if len(parts) >= 4:
            db = parts[1]
            machine_type = parts[2]
            machine_id_with_file = parts[3]
            machine_id = machine_id_with_file.split('-')[0] if '-' in machine_id_with_file else machine_id
            
            key = (db, machine_type, machine_id)
            
            if not is_normal:  # We're processing abnormal dir initially, so these are normal files
                if key not in normal_data:
                    normal_data[key] = []
                normal_data[key].append(str(file_path))
            else:  # We're processing normal dir initially, so these are abnormal files
                if key not in abnormal_data:
                    abnormal_data[key] = []
                abnormal_data[key].append(str(file_path))
    
    # Check if we have any files
    if not normal_data:
        logger.warning(f"No normal {ext} files found")
        if not abnormal_data:
            logger.error(f"No files found at all")
            return [], [], [], [], [], []
    
    if not abnormal_data:
        logger.warning(f"No abnormal {ext} files found")
    
    # Apply filter if enabled
    if param.get("filter", {}).get("enabled", False):
        filter_db = param["filter"].get("db_level")
        filter_machine = param["filter"].get("machine_type")
        filter_id = param["filter"].get("machine_id")
        
        filtered_normal_data = {}
        filtered_abnormal_data = {}
        
        for key in normal_data:
            db, machine_type, machine_id = key
            if (not filter_db or db == filter_db) and \
               (not filter_machine or machine_type == filter_machine) and \
               (not filter_id or machine_id == filter_id):
                filtered_normal_data[key] = normal_data[key]
        
        for key in abnormal_data:
            db, machine_type, machine_id = key
            if (not filter_db or db == filter_db) and \
               (not filter_machine or machine_type == filter_machine) and \
               (not filter_id or machine_id == filter_id):
                filtered_abnormal_data[key] = abnormal_data[key]
        
        normal_data = filtered_normal_data
        abnormal_data = filtered_abnormal_data
    
    # Get all normal and abnormal files
    normal_files = []
    for files in normal_data.values():
        normal_files.extend(files)
    
    abnormal_files = []
    for files in abnormal_data.values():
        abnormal_files.extend(files)
    
    # Create labels
    normal_labels = np.zeros(len(normal_files))
    abnormal_labels = np.ones(len(abnormal_files))
    
    # Print the number of files
    num_normal = len(normal_files)
    num_abnormal = len(abnormal_files)
    logger.info(f"Number of normal samples: {num_normal}")
    logger.info(f"Number of abnormal samples: {num_abnormal}")
    logger.info(f"Total samples: {num_normal + num_abnormal}")
    
    # Shuffle files while keeping labels aligned
    if num_normal > 0:
        normal_indices = np.arange(len(normal_files))
        np.random.shuffle(normal_indices)
        normal_files = [normal_files[i] for i in normal_indices]
        normal_labels = normal_labels[normal_indices]
    
    if num_abnormal > 0:
        abnormal_indices = np.arange(len(abnormal_files))
        np.random.shuffle(abnormal_indices)
        abnormal_files = [abnormal_files[i] for i in abnormal_indices]
        abnormal_labels = abnormal_labels[abnormal_indices]
    
    # Calculate split indices
    n_normal_train = int(num_normal * split_ratio[0])
    n_normal_val = int(num_normal * split_ratio[1])
    n_abnormal_train = int(num_abnormal * split_ratio[0])
    n_abnormal_val = int(num_abnormal * split_ratio[1])
    
    # Split normal files
    normal_train_files = normal_files[:n_normal_train] if num_normal > 0 else []
    normal_train_labels = normal_labels[:n_normal_train] if num_normal > 0 else np.array([])
    normal_val_files = normal_files[n_normal_train:n_normal_train+n_normal_val] if num_normal > 0 else []
    normal_val_labels = normal_labels[n_normal_train:n_normal_train+n_normal_val] if num_normal > 0 else np.array([])
    normal_test_files = normal_files[n_normal_train+n_normal_val:] if num_normal > 0 else []
    normal_test_labels = normal_labels[n_normal_train+n_normal_val:] if num_normal > 0 else np.array([])
    
    # Split abnormal files
    abnormal_train_files = abnormal_files[:n_abnormal_train] if num_abnormal > 0 else []
    abnormal_train_labels = abnormal_labels[:n_abnormal_train] if num_abnormal > 0 else np.array([])
    abnormal_val_files = abnormal_files[n_abnormal_train:n_abnormal_train+n_abnormal_val] if num_abnormal > 0 else []
    abnormal_val_labels = abnormal_labels[n_abnormal_train:n_abnormal_train+n_abnormal_val] if num_abnormal > 0 else np.array([])
    abnormal_test_files = abnormal_files[n_abnormal_train+n_abnormal_val:] if num_abnormal > 0 else []
    abnormal_test_labels = abnormal_labels[n_abnormal_train+n_abnormal_val:] if num_abnormal > 0 else np.array([])


    if len(normal_test_files) == 0:
        logger.warning("No normal samples in test set! Adding some from training set.")
        # Move some normal samples from train to test if available
        move_count = min(5, len(normal_train_files))
        if move_count > 0:
            normal_test_files = normal_train_files[:move_count]
            normal_test_labels = normal_train_labels[:move_count]
            normal_train_files = normal_train_files[move_count:]
            normal_train_labels = normal_train_labels[move_count:]

    if len(abnormal_test_files) == 0:
        logger.warning("No abnormal samples in test set! Adding some from training set.")
        # Move some abnormal samples from train to test if available
        move_count = min(5, len(abnormal_train_files))
        if move_count > 0:
            abnormal_test_files = abnormal_train_files[:move_count]
            abnormal_test_labels = abnormal_train_labels[:move_count]
            abnormal_train_files = abnormal_train_files[move_count:]
            abnormal_train_labels = abnormal_train_labels[move_count:]
    
    # Combine normal and abnormal datasets
    train_files = normal_train_files + abnormal_train_files
    train_labels = np.concatenate([normal_train_labels, abnormal_train_labels]) if len(normal_train_labels) > 0 or len(abnormal_train_labels) > 0 else np.array([])
    val_files = normal_val_files + abnormal_val_files
    val_labels = np.concatenate([normal_val_labels, abnormal_val_labels]) if len(normal_val_labels) > 0 or len(abnormal_val_labels) > 0 else np.array([])
    test_files = normal_test_files + abnormal_test_files
    test_labels = np.concatenate([normal_test_labels, abnormal_test_labels]) if len(normal_test_labels) > 0 or len(abnormal_test_labels) > 0 else np.array([])
    
    # Shuffle the training, validation and test sets
    if len(train_files) > 0:
        train_indices = np.arange(len(train_files))
        np.random.shuffle(train_indices)
        train_files = [train_files[i] for i in train_indices]
        train_labels = train_labels[train_indices]
    
    if len(val_files) > 0:
        val_indices = np.arange(len(val_files))
        np.random.shuffle(val_indices)
        val_files = [val_files[i] for i in val_indices]
        val_labels = val_labels[val_indices]
    
    if len(test_files) > 0:
        test_indices = np.arange(len(test_files))
        np.random.shuffle(test_indices)
        test_files = [test_files[i] for i in test_indices]
        test_labels = test_labels[test_indices]
    
    logger.info(f"train_file num : {len(train_files)} (normal: {len(normal_train_files)}, abnormal: {len(abnormal_train_files)})")
    logger.info(f"val_file num : {len(val_files)} (normal: {len(normal_val_files)}, abnormal: {len(abnormal_val_files)})")
    logger.info(f"test_file num : {len(test_files)} (normal: {len(normal_test_files)}, abnormal: {len(abnormal_test_files)})")

    # Debug info
    print_with_timestamp(f"Looking for files in: {target_dir}")
    print_with_timestamp(f"Found {len(files_in_dir)} files")
    print_with_timestamp(f"DEBUG - Dataset summary:")
    print_with_timestamp(f"  Normal files found: {len(normal_files)}")
    print_with_timestamp(f"  Abnormal files found: {len(abnormal_files)}")
    print_with_timestamp(f"  Normal train: {len(normal_train_files)}, Normal val: {len(normal_val_files)}, Normal test: {len(normal_test_files)}")
    print_with_timestamp(f"  Abnormal train: {len(abnormal_train_files)}, Abnormal val: {len(abnormal_val_files)}, Abnormal test: {len(abnormal_test_files)}")

    return train_files, train_labels, val_files, val_labels, test_files, test_labels

########################################################################
# main
########################################################################

def main():
    # Set memory growth before any other TensorFlow operations
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                # Enable memory growth for better memory management
                tf.config.experimental.set_memory_growth(device, True)
                logger.info(f"Enabled memory growth for {device}")
            except Exception as e:
                logger.warning(f"Could not set memory growth for {device}: {e}")
        
        # Verify GPU is being used
        logger.info(f"TensorFlow is using GPU: {tf.test.is_gpu_available()}")
        logger.info(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")
    
    # Configure memory growth for V100 32GB
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # ONLY enable memory growth without setting a fixed limit
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Remove the memory_limit configuration completely
            logger.info("GPU memory growth enabled - will allocate memory as needed")
        except RuntimeError as e:
            logger.error(f"Error setting GPU memory configuration: {e}")

    # Load configuration file (fixed path with single underscore)
    config_file = "baseline_AST.yaml"
    print_with_timestamp(f"Loading configuration from {config_file}")
    
    try:
        with open(config_file, "r") as stream:
            param = yaml.safe_load(stream)
        print_with_timestamp("Configuration loaded successfully")
    except Exception as e:
        print_with_timestamp(f"Error loading configuration: {e}")
        print_with_timestamp("Trying alternative configuration filename...")
        try:
            alt_config_file = "baseline__AST.yaml"
            with open(alt_config_file, "r") as stream:
                param = yaml.safe_load(stream)
            print_with_timestamp(f"Configuration loaded successfully from {alt_config_file}")
        except Exception as e2:
            print_with_timestamp(f"Failed to load configuration from alternative file: {e2}")
            print_with_timestamp("Please make sure 'baseline_AST.yaml' exists in the current directory")
            return

    # Enable XLA compilation for faster GPU execution
    if param.get("training", {}).get("xla_acceleration", True):
        logger.info("Enabling XLA acceleration for faster training")
        try:
            tf.config.optimizer.set_jit(True)  # Enable XLA
            os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'  # Force XLA on all operations
            logger.info("XLA acceleration enabled successfully")
        except Exception as e:
            logger.warning(f"Failed to enable XLA acceleration: {e}")


    print_with_timestamp("============== CHECKING DIRECTORY STRUCTURE ==============")
    normal_dir = Path(param["base_directory"]) / "normal"
    abnormal_dir = Path(param["base_directory"]) / "abnormal"

    print_with_timestamp(f"Normal directory exists: {normal_dir.exists()}")
    if normal_dir.exists():
        normal_files = list(normal_dir.glob("*.wav"))
        print_with_timestamp(f"Number of normal files found: {len(normal_files)}")
        if normal_files:
            print_with_timestamp(f"Sample normal filename: {normal_files[0].name}")

    print_with_timestamp(f"Abnormal directory exists: {abnormal_dir.exists()}")
    if abnormal_dir.exists():
        abnormal_files = list(abnormal_dir.glob("*.wav"))
        print_with_timestamp(f"Number of abnormal files found: {len(abnormal_files)}")
        if abnormal_files:
            print_with_timestamp(f"Sample abnormal filename: {abnormal_files[0].name}")

    start_time = time.time()

    os.makedirs(param["pickle_directory"], exist_ok=True)
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["result_directory"], exist_ok=True)

    visualizer = Visualizer(param)

    # Test audio file loading directly
    normal_path = Path(param["base_directory"]) / "normal"
    abnormal_path = Path(param["base_directory"]) / "abnormal"
    
    # Try to load a sample file directly
    test_files = list(normal_path.glob("*.wav"))[:1] if normal_path.exists() and list(normal_path.glob("*.wav")) else list(abnormal_path.glob("*.wav"))[:1]
    
    if test_files:
        print_with_timestamp(f"DEBUG: Testing direct audio load for: {test_files[0]}")
        sr, y = demux_wav(str(test_files[0]))
        if y is not None:
            print_with_timestamp(f"DEBUG: Successfully loaded audio with sr={sr}, length={len(y)}")
        else:
            print_with_timestamp(f"DEBUG: Failed to load audio file")
    else:
        print_with_timestamp("DEBUG: No test files found to verify audio loading")

    base_path = Path(param["base_directory"])

    print_with_timestamp("============== COUNTING DATASET SAMPLES ==============")
    logger.info("Counting all samples in the dataset...")
    
    normal_path = Path(param["base_directory"]) / "normal"
    abnormal_path = Path(param["base_directory"]) / "abnormal"
    
    # Count files
    total_normal_files = len(list(normal_path.glob("*.wav"))) if normal_path.exists() else 0
    total_abnormal_files = len(list(abnormal_path.glob("*.wav"))) if abnormal_path.exists() else 0
    
    # Log the total counts
    logger.info(f"Total normal files in dataset: {total_normal_files}")
    logger.info(f"Total abnormal files in dataset: {total_abnormal_files}")
    logger.info(f"Total files in dataset: {total_normal_files + total_abnormal_files}")

    # Define target_dir (use normal_path as default, since dataset_generator will find corresponding abnormal files)
    target_dir = normal_path if normal_path.exists() else abnormal_path
    if not target_dir.exists():
        logger.error("Neither normal nor abnormal directory exists!")
        return

    # Get model file information from the first file in the directory
    sample_files = list(Path(target_dir).glob(f"*.{param.get('dataset', {}).get('file_extension', 'wav')}"))
    print_with_timestamp(f"DEBUG: Found {len(sample_files)} files in {target_dir}")
    print_with_timestamp(f"DEBUG: First 5 files: {[f.name for f in sample_files[:5]]}")

    if not sample_files:
        logger.warning(f"No files found in {target_dir}")
        return  # Exit main() if no files are found

    # Parse a sample filename to get db, machine_type, machine_id
    filename = sample_files[0].name
    parts = filename.split('_')
    print_with_timestamp(f"DEBUG: Parsing filename '{filename}' into parts: {parts}")

    if len(parts) < 4:
        logger.warning(f"Filename format incorrect: {filename}")
        return  # Exit main() if filename format is incorrect
    
    # Use a straightforward key without unintended characters
    evaluation_result_key = "overall_model"
    print_with_timestamp(f"DEBUG: Using evaluation_result_key: {evaluation_result_key}")

    # Initialize evaluation result dictionary
    evaluation_result = {}
    
    # Initialize results dictionary if it doesn't exist
    results = {}
    all_y_true = []
    all_y_pred = []
    result_file = f"{param['result_directory']}/result__AST.yaml"

    # Get chunking parameters - add these variables that were missing
    chunking_enabled = param.get("feature", {}).get("dataset_chunking", {}).get("enabled", False)
    chunk_size = param.get("feature", {}).get("dataset_chunking", {}).get("chunk_size", 5000)
    preprocessing_batch_size = param.get("feature", {}).get("preprocessing_batch_size", 64)
    
    print_with_timestamp("============== DATASET_GENERATOR ==============")
    
    # Generate dataset
    train_files, train_labels, val_files, val_labels, test_files, test_labels = dataset_generator(target_dir, param=param)
    
    # Verify generated dataset
    logger.info(f"Generated train files: {len(train_files)}")
    logger.info(f"Generated val files: {len(val_files)}")
    logger.info(f"Generated test files: {len(test_files)}")
    
    # Check if dataset generation was successful
    if len(train_files) == 0 or len(val_files) == 0 or len(test_files) == 0:
        logger.error(f"No files found for {evaluation_result_key}, skipping...")
        return  # Exit main() if no files are found after generation
    
    # After dataset generation
    # ... existing code for dataset_generator and verification ...
    
    # Simplified main flow with clear progress reporting
    print_with_timestamp("\n============== STARTING DATA PROCESSING ==============")
    
    # Define pickle file paths
    train_pickle = f"{param['pickle_directory']}/train_{evaluation_result_key}.pickle"
    train_labels_pickle = f"{param['pickle_directory']}/train_labels_{evaluation_result_key}.pickle"
    val_pickle = f"{param['pickle_directory']}/val_{evaluation_result_key}.pickle"
    val_labels_pickle = f"{param['pickle_directory']}/val_labels_{evaluation_result_key}.pickle"
    test_files_pickle = f"{param['pickle_directory']}/test_files_{evaluation_result_key}.pickle"
    test_labels_pickle = f"{param['pickle_directory']}/test_labels_{evaluation_result_key}.pickle"
    
    # Check if preprocessed data already exists
    if (os.path.exists(train_pickle) and os.path.exists(train_labels_pickle) and
        os.path.exists(val_pickle) and os.path.exists(val_labels_pickle) and
        os.path.exists(test_files_pickle) and os.path.exists(test_labels_pickle)):
        
        print_with_timestamp("Found existing preprocessed data. Loading from pickle files...")
        
        try:
            train_data = load_pickle(train_pickle)
            train_labels_expanded = load_pickle(train_labels_pickle)
            val_data = load_pickle(val_pickle)
            val_labels_expanded = load_pickle(val_labels_pickle)
            test_files = load_pickle(test_files_pickle)
            test_labels_expanded = load_pickle(test_labels_pickle)
            
            print_with_timestamp(f"Loaded train data shape: {train_data.shape}")
            print_with_timestamp(f"Loaded validation data shape: {val_data.shape}")
            print_with_timestamp(f"Loaded test files: {len(test_files)}")
            
        except Exception as e:
            print_with_timestamp(f"Error loading pickle data: {e}")
            print_with_timestamp("Will generate new preprocessed data...")
            train_data = None  # Reset to trigger data generation
    else:
        print_with_timestamp("No preprocessed data found. Will generate new data...")
        train_data = None
    
    # Process data if needed (wasn't loaded from pickle)
    if train_data is None:
        print_with_timestamp("Processing training data...")
        train_data, train_labels_expanded = list_to_spectrograms(
            train_files, train_labels, msg="Generating train dataset", 
            augment=False, param=param, batch_size=preprocessing_batch_size
        )
        
        print_with_timestamp("Processing validation data...")
        val_data, val_labels_expanded = list_to_spectrograms(
            val_files, val_labels, msg="Generating validation dataset", 
            augment=False, param=param, batch_size=preprocessing_batch_size
        )
        
        test_labels_expanded = test_labels
        
        # Save processed data
        print_with_timestamp("Saving processed data to pickle files...")
        try:
            save_pickle(train_pickle, train_data)
            save_pickle(train_labels_pickle, train_labels_expanded)
            save_pickle(val_pickle, val_data)
            save_pickle(val_labels_pickle, val_labels_expanded)
            save_pickle(test_files_pickle, test_files)
            save_pickle(test_labels_pickle, test_labels_expanded)
            print_with_timestamp("Successfully saved processed data")
        except Exception as e:
            print_with_timestamp(f"Error saving processed data: {e}")
    
    # Print data summary
    print_with_timestamp("\n============== DATA SUMMARY ==============")
    print_with_timestamp(f"Training data: {train_data.shape}, labels: {train_labels_expanded.shape}")
    print_with_timestamp(f"Validation data: {val_data.shape}, labels: {val_labels_expanded.shape}")
    print_with_timestamp(f"Test files: {len(test_files)}")
    
    # Configure mixed precision
    print_with_timestamp("\n============== CONFIGURING MODEL ==============")
    if param.get("training", {}).get("mixed_precision", True):
        print_with_timestamp("Enabling mixed precision training")
        mixed_precision.set_global_policy('mixed_float16')
        print_with_timestamp(f"Mixed precision policy: {mixed_precision.global_policy()}")
    
    # Define target shape for model input
    target_shape = (param["feature"]["n_mels"], 96)  # Default shape
    print_with_timestamp(f"Using target shape: {target_shape}")
    
    # Create model
    print_with_timestamp("Creating AST model...")
    model = create_ast_model(
        input_shape=target_shape,
        config=param.get("model", {}).get("architecture", {})
    )
    
    # Print model summary
    print_with_timestamp("Model architecture:")
    model.summary(print_fn=lambda x: print_with_timestamp(x))
    
    # Compile model
    print_with_timestamp("Compiling model...")
    model.compile(
        optimizer=AdamW(
            learning_rate=param.get("fit", {}).get("compile", {}).get("learning_rate", 0.0001),
            weight_decay=0.01,
            clipnorm=param.get("training", {}).get("gradient_clip_norm", 1.0)
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Setup callbacks
    print_with_timestamp("Setting up training callbacks...")
    callbacks = []
    
    # Early stopping
    if param.get("fit", {}).get("early_stopping", {}).get("enabled", True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=param.get("fit", {}).get("early_stopping", {}).get("monitor", "val_loss"),
                patience=param.get("fit", {}).get("early_stopping", {}).get("patience", 15),
                min_delta=param.get("fit", {}).get("early_stopping", {}).get("min_delta", 0.001),
                restore_best_weights=param.get("fit", {}).get("early_stopping", {}).get("restore_best_weights", True)
            )
        )
        print_with_timestamp("Added early stopping callback")
    
    # Learning rate scheduler
    if param.get("fit", {}).get("lr_scheduler", {}).get("enabled", True):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=param.get("fit", {}).get("lr_scheduler", {}).get("monitor", "val_loss"),
                factor=param.get("fit", {}).get("lr_scheduler", {}).get("factor", 0.1),
                patience=param.get("fit", {}).get("lr_scheduler", {}).get("patience", 5),
                min_delta=param.get("fit", {}).get("lr_scheduler", {}).get("min_delta", 0.001),
                cooldown=param.get("fit", {}).get("lr_scheduler", {}).get("cooldown", 2),
                min_lr=param.get("fit", {}).get("lr_scheduler", {}).get("min_lr", 0.00000001)
            )
        )
        print_with_timestamp("Added learning rate scheduler callback")
    
    # Model checkpointing
    if param.get("fit", {}).get("checkpointing", {}).get("enabled", True):
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{param['model_directory']}/best_model.keras",
                monitor=param.get("fit", {}).get("checkpointing", {}).get("monitor", "val_accuracy"),
                mode=param.get("fit", {}).get("checkpointing", {}).get("mode", "max"),
                save_best_only=param.get("fit", {}).get("checkpointing", {}).get("save_best_only", True)
            )
        )
        print_with_timestamp("Added model checkpoint callback")
    
    # Create TensorFlow datasets
    print_with_timestamp("\n============== CREATING TENSORFLOW DATASETS ==============")
    batch_size = param.get("fit", {}).get("batch_size", 8)
    
    print_with_timestamp(f"Creating training dataset with batch size {batch_size}...")
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels_expanded))
    train_dataset = train_dataset.batch(batch_size).shuffle(1000).prefetch(tf.data.AUTOTUNE)
    
    print_with_timestamp("Creating validation dataset...")
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels_expanded))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Start training
    print_with_timestamp("\n============== STARTING MODEL TRAINING ==============")
    print_with_timestamp(f"Training for {param['fit']['epochs']} epochs...")
    
    history = model.fit(
        train_dataset,
        epochs=param["fit"]["epochs"],
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save trained model
    print_with_timestamp("\n============== SAVING MODEL ==============")
    try:
        model.save(f"{param['model_directory']}/final_model.keras")
        print_with_timestamp("Model saved successfully!")
    except Exception as e:
        print_with_timestamp(f"Error saving model: {e}")
    
    # Plot training history
    print_with_timestamp("\n============== VISUALIZING RESULTS ==============")
    visualizer.loss_plot(history)
    visualizer.save_figure(f"{param['result_directory']}/training_history.png")
    print_with_timestamp(f"Training history plot saved to {param['result_directory']}/training_history.png")
    
    # Calculate training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print_with_timestamp(f"\nTotal training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    print_with_timestamp("\n============== TRAINING COMPLETE ==============")
    return