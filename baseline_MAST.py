#!/usr/bin/env python
"""
 @file   baseline_MAST.py
 @brief  Masked Audio Spectrogram Transformer (MAST) implementation for anomaly detection
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
import hashlib
import optuna
import subprocess
from numba import cuda

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, Add, MultiHeadAttention, LayerNormalization, Reshape, Permute, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.losses import mse as mean_squared_error
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from skimage.metrics import structural_similarity as ssim
from transformers import TFViTModel, ViTConfig
from tensorflow.keras import layers
########################################################################

########################################################################
# version
########################################################################
__versions__ = "3.0.0"
########################################################################

# Load configuration from YAML
with open("baseline_MAST.yaml", "r") as yaml_file:
    config = yaml.safe_load(yaml_file)

# Clear GPU memory if enabled in config
if config.get("clear_gpu_memory", False):
    def clear_gpu_memory():
        """Clear GPU memory to prevent OOM errors."""
        tf.keras.backend.clear_session()
        gc.collect()
        try:
            cuda.select_device(0)
            cuda.close()
        except Exception as e:
            logger.warning(f"Failed to clear GPU memory: {e}")

    clear_gpu_memory()

# Focal loss function
# Modify to accept gamma and alpha as parameters
def focal_loss(gamma, alpha):
    def loss_function(y_true, y_pred):
        dtype = y_pred.dtype
        y_true = tf.cast(y_true, dtype)
        dtype = y_pred.dtype
        y_true = tf.cast(y_true, dtype)
        eps = tf.cast(tf.keras.backend.epsilon(), dtype)
        gamma_c = tf.cast(gamma, dtype)
        alpha_c = tf.cast(alpha, dtype)
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = tf.where(tf.equal(y_true, tf.constant(1, dtype=dtype)), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma_c)
        alpha_weight = tf.where(tf.equal(y_true, tf.constant(1, dtype=dtype)), alpha_c, 1 - alpha_c)
        loss = alpha_weight * focal_weight * cross_entropy
        return tf.reduce_mean(loss)

    return loss_function

# Positional encoding function
def positional_encoding(seq_len, d_model):
    encoding_type = config["positional_encoding"]["encoding_type"]
    positions = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    pe = np.expand_dims(pe, axis=0)
    return tf.cast(pe, dtype=tf.float32)

########################################################################
# setup STD I/O
########################################################################
def setup_logging():
    os.makedirs("./logs/log_MAST", exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, filename="./logs/log_MAST/baseline_MAST.log")
    logger = logging.getLogger(' ')
    
    # Clear existing handlers to prevent duplicate messages
    if logger.handlers:
        logger.handlers.clear()
        
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
            
            if frames and stride:
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
            
        return log_mel_spectrogram
        
    except Exception as e:
        logger.error(f"Error in file_to_spectrogram for {file_name}: {e}")
        return None



def list_to_spectrograms(file_list, labels=None, msg="calc...", augment=False, param=None, batch_size=16, use_cache=True):
    """
    Process a list of files into spectrograms with optional labels - memory optimized version with caching
    """
    n_mels = param.get("feature", {}).get("n_mels", 64)
    n_fft = param.get("feature", {}).get("n_fft", 1024)
    hop_length = param.get("feature", {}).get("hop_length", 512)
    power = param.get("feature", {}).get("power", 2.0)
    
    # Ensure use_cache is defined and accessible
    use_cache = param.get("cache", {}).get("enabled", True) if param else True
    
    # Check if caching is enabled in config
    use_cache = param.get("cache", {}).get("enabled", True) if param else True
    
    # Add cache metrics tracking
    cache_hits = 0
    cache_misses = 0

    # First pass: determine dimensions and count valid files
    valid_files = []
    valid_labels = [] if labels is not None else None
    max_freq = 0
    max_time = 0
    
    logger.info(f"First pass: checking dimensions of {len(file_list)} files")
    for idx, file_path in enumerate(tqdm(file_list, desc=f"{msg} (dimension check)")):
        try:
            # Use cached version for dimension check if enabled and not augmenting
            spec = cached_file_to_spectrogram(file_path, n_mels, n_fft, hop_length, power, False, param)
            if spec is not None:
                cache_hits += 1
                valid_files.append(file_path)
                if labels is not None:
                    valid_labels.append(labels[idx])
                max_freq = max(max_freq, spec.shape[0])
                max_time = max(max_time, spec.shape[1])
                continue
            else:
                cache_misses += 1
                
            # Use cached version for dimension check if enabled and not augmenting
            if use_cache and not augment:
                spec = cached_file_to_spectrogram(file_path, n_mels, n_fft, hop_length, power, False, param)
            else:
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
    
    # Handle division by zero in cache performance logging
    if cache_hits + cache_misses > 0:
        logger.info(f"Cache performance: {cache_hits} hits, {cache_misses} misses, {cache_hits / (cache_hits + cache_misses) * 100:.1f}% hit rate")
    else:
        logger.info("Cache performance: No cache hits or misses recorded.")

    # Use target shape from parameters if available
    target_shape = param.get("feature", {}).get("target_shape", None)
    if target_shape:
        max_freq, max_time = target_shape
    else:
        # Round up to nearest multiple of 8 for better GPU utilization
        max_freq = ((max_freq + 7) // 8) * 8
        max_time = ((max_time + 7) // 8) * 8
    
    logger.info(f"Using target shape: ({max_freq}, {max_time})")
    
    # Second pass: process files in batches with caching
    total_valid = len(valid_files)
    spectrograms = np.zeros((total_valid, max_freq, max_time), dtype=np.float32)
    processed_labels = np.array(valid_labels) if valid_labels else None
    
    for batch_start in tqdm(range(0, total_valid, batch_size), desc=f"{msg} (processing)"):
        batch_end = min(batch_start + batch_size, total_valid)
        batch_files = valid_files[batch_start:batch_end]
        
        for i, file_path in enumerate(batch_files):
            try:
                # Use cached version if enabled and not augmenting
                if use_cache and not augment:
                    spec = cached_file_to_spectrogram(file_path, n_mels, n_fft, hop_length, power, False, param)
                else:
                    spec = file_to_spectrogram(file_path, n_mels, n_fft, hop_length, power, augment, param)
                
                if spec is not None:
                    # Handle 3D input
                    if len(spec.shape) == 3:
                        spec = spec[0]
                    
                    # Resize if needed
                    if spec.shape[0] != max_freq or spec.shape[1] != max_time:
                        try:
                            from skimage.transform import resize
                            spec = resize(spec, (max_freq, max_time), mode='reflect', anti_aliasing=True)
                        except ImportError:
                            logger.error("scikit-image is required for resizing spectrograms")
                    
                    # Store in output array
                    spectrograms[batch_start + i] = spec
                    
                    # Clear memory
                    del spec
                    
            except tf.errors.ResourceExhaustedError:
                logger.warning("CUDA OOM detected. Reducing batch size dynamically.")
                batch_size = max(1, batch_size // 2)
                gc.collect()
                tf.keras.backend.clear_session()
                return list_to_spectrograms(file_list, labels, msg, augment, param, batch_size)

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                # Fill with zeros for failed files
                spectrograms[batch_start + i] = np.zeros((max_freq, max_time))
        
        gc.collect()

    spectrograms = spectrograms.astype(np.float32)
    
    if labels is not None:
        return spectrograms, processed_labels
    return spectrograms



def preprocess_spectrograms(spectrograms, target_shape, param=None, use_cache=False):
    """
    Resize all spectrograms to a consistent shape.
    """
    # Handle case where input is a list of file paths instead of spectrograms
    if isinstance(spectrograms, list):
        logger.info(f"Converting {len(spectrograms)} file paths to spectrograms...")
        # Pass the param and use_cache to the list_to_spectrograms function
        result = list_to_spectrograms(spectrograms, None, "Processing files", False, param, use_cache=use_cache)

        # Check if the result is a tuple and unpack accordingly
        if isinstance(result, tuple):
            if len(result) >= 2:
                spectrograms, _ = result[:2]
            else:
                spectrograms = result[0]
        else:
            spectrograms = result

    # Handle empty spectrograms
    if spectrograms is None or spectrograms.shape[0] == 0 or spectrograms.shape[1] == 0 or spectrograms.shape[2] == 0:
        logger.error("No valid spectrograms generated. Please check the input data and parameters.")
        return None, None

    batch_size = spectrograms.shape[0]
    processed = np.zeros((batch_size, target_shape[0], target_shape[1]), dtype=np.float32)

    for i in range(batch_size):
        # Get current spectrogram
        spec = spectrograms[i]

        # Handle 3D input (when a single frame has an extra dimension)
        if len(spec.shape) == 3 and spec.shape[2] == 1:
            spec = spec[:, :, 0]  # Remove the last dimension

        # Skip if dimensions already match
        if spec.shape[0] == target_shape[0] and spec.shape[1] == target_shape[1]:
            processed[i] = spec
            continue

        try:
            # Simple resize using interpolation
            from skimage.transform import resize
            resized_spec = resize(spec, target_shape, anti_aliasing=True, mode='reflect')
            processed[i] = resized_spec
        except Exception as e:
            logger.error(f"Error resizing spectrogram: {e}")
            # If resize fails, use simple padding/cropping
            temp_spec = np.zeros(target_shape)
            # Copy as much as will fit
            freq_dim = min(spec.shape[0], target_shape[0])
            time_dim = min(spec.shape[1], target_shape[1])
            temp_spec[:freq_dim, :time_dim] = spec[:freq_dim, :time_dim]
            processed[i] = temp_spec

    return processed


########################################################################
# GPU monitoring
########################################################################
def monitor_gpu_usage():
    """
    Monitor GPU memory usage and return used memory, total memory, and usage percentage.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        logger.warning("No GPUs detected. Using CPU instead.")
        return 0, 0, 0.0

    try:
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        used_mem = memory_info['current'] // (1024 * 1024)  # Convert to MB
        total_mem = memory_info['peak'] // (1024 * 1024)  # Convert to MB
        usage_pct = (used_mem / total_mem) * 100 if total_mem > 0 else 0.0
        return used_mem, total_mem, usage_pct
    except Exception as e:
        logger.error(f"Failed to monitor GPU usage: {e}")
        return 0, 0, 0.0

########################################################################
# Mixed precision configuration
########################################################################
def configure_mixed_precision(enable=True):
    """
    Configure mixed precision training if enabled and supported by the hardware.
    """
    if enable:
        try:
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
            logger.info("Mixed precision training enabled.")
        except ImportError as e:
            logger.warning(f"Failed to enable mixed precision: {e}")
    else:
        logger.info("Mixed precision training is disabled.")

########################################################################
# Dataset generator
########################################################################
def dataset_generator(base_dir, config):
    """
    Placeholder function to load and split the dataset into training, validation, and test sets.
    Args:
        base_dir (str): Base directory containing the dataset.
        config (dict): Configuration dictionary.

    Returns:
        tuple: Train files, train labels, validation files, validation labels, test files, test labels.
    """
    from sklearn.model_selection import train_test_split
    import glob

    # Load all files with the specified extension
    file_extension = config.get('dataset', {}).get('file_extension', 'wav')
    all_files = glob.glob(f"{base_dir}/**/*.{file_extension}", recursive=True)

    # Generate dummy labels (0 for normal, 1 for abnormal)
    labels = [0 if 'normal' in file else 1 for file in all_files]

    # Split into train, validation, and test sets
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )

    return train_files, train_labels, val_files, val_labels, test_files, test_labels

########################################################################
# main
########################################################################
def main():
    exec_start = time.time()
    # Enable dynamic GPU memory growth and cap usage
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=30000)]  # Adjusted to fit within 32GB GPU memory
        )
    # Print current VRAM usage to verify GPU memory setup
    used_mem, total_mem, usage_pct = monitor_gpu_usage()
    logger.info(f"Initial GPU memory usage: {used_mem}MB/{total_mem}MB ({usage_pct:.1f}%)")
    # Load configurations
    with open('baseline_MAST.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Convert relative paths to absolute paths for VPS
    base_dir = os.path.abspath(config.get('base_directory', './dataset'))
    model_dir = os.path.abspath(config.get('model_directory', './model/MAST'))
    result_dir = os.path.abspath(config.get('result_directory', './result/result_MAST'))
    model_path = os.path.join(model_dir, 'mast_model.keras')

    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs('pickle/pickle_mast', exist_ok=True)

    # Extract configurations
    model_params = config.get('model', {})
    mast_params = config.get('mast', {})
    transformer_params = config.get('model', {}).get('architecture', {}).get('transformer', {})
    dataset_params = config.get('dataset', {})
    training_params = config.get('training', {})
    
    # Set up logging
    setup_logging()
    
    # Log configuration info
    logger.info(f"Starting MAST model training with config: {config}")
    
    # Enable optimized training modes
    # Mixed precision for V100 GPU
    configure_mixed_precision(training_params.get('mixed_precision', True))
    # Enable XLA if configured
    if training_params.get('xla_acceleration', False):
        logger.info("Enabling XLA JIT compilation")
        tf.config.optimizer.set_jit(True)
    # Check if we should load existing model or create a new one
    if training_params.get('load_model', False) and os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        # Set random seeds for reproducibility
        tf.random.set_seed(training_params.get('random_seed', 42))
        np.random.seed(training_params.get('random_seed', 42))
        
        # Load and preprocess dataset
        logger.info("Loading dataset")
        # Fix: Use the normal directory to properly process both normal and abnormal data
        normal_dir = os.path.join(base_dir, 'normal')
        
        train_files, train_labels, val_files, val_labels, test_files, test_labels = dataset_generator(
            normal_dir, config)
        
        # Get input shape from data
        target_shape = (config["feature"]["n_mels"], 96)
        logger.info(f"Target spectrogram shape: {target_shape}")
        
        # Preprocess to ensure consistent shapes
        logger.info("Preprocessing training data...")
        # Pass the config to preprocess_spectrograms
        train_data = preprocess_spectrograms(train_files, target_shape, config, use_cache=True)

        # Fix AttributeError by checking train_data
        if train_data is None or isinstance(train_data, tuple):
            logger.error("Train data is invalid. Please check preprocessing.")
            return

        logger.info(f"Preprocessed train data shape: {train_data.shape}")

        logger.info("Preprocessing validation data...")
        # Pass the config to preprocess_spectrograms
        val_data = preprocess_spectrograms(val_files, target_shape, config, use_cache=True)
        logger.info(f"Preprocessed validation data shape: {val_data.shape}")
        
        # Normalize data for better training
        logger.info("Normalizing data...")
        # Calculate mean and std from training data
        train_mean = np.mean(train_data)
        train_std = np.std(train_data)
        logger.info(f"Training data statistics - Mean: {train_mean:.4f}, Std: {train_std:.4f}")

        # Apply normalization if std is not too small
        if train_std > 1e-6:
            train_data = (train_data - train_mean) / train_std
            val_data = (val_data - train_mean) / train_std
            logger.info("Z-score normalization applied")
        else:
            # If std is too small, just center the data
            train_data = train_data - train_mean
            val_data = val_data - train_mean
            logger.info("Mean centering applied (std too small for z-score)")

        # Balance the dataset
        logger.info("Balancing dataset...")
        train_data, train_labels_expanded = balance_dataset(train_data, train_labels, augment_minority=True)
        
        # Create pretrain and finetune models
        pretrain_model, finetune_model = create_mast_model(target_shape, mast_params, transformer_params)
        
        # Check if we should perform pretraining
        if mast_params.get('pretraining', {}).get('enabled', True):
            logger.info("Starting MAST pretraining phase with masking")
            
            # Configure pretraining parameters
            pretrain_epochs = mast_params.get('pretraining', {}).get('epochs', 50)
            pretrain_batch_size = mast_params.get('pretraining', {}).get('batch_size', 32)
            # Cast learning rate from config to float (handles strings like '1e-4')
            pretrain_lr_raw = mast_params.get('pretraining', {}).get('learning_rate', 1e-4)
            pretrain_lr = float(pretrain_lr_raw)
            
            # Prepare data for pretraining (no labels needed, just the spectrograms)
            # Combine all available data for pretraining
            X_pretrain = np.concatenate([train_data, val_data], axis=0)
            
            # Apply masking to create input-target pairs for reconstruction
            mask_probability = mast_params.get('pretraining', {}).get('masking', {}).get('probability', 0.15)
            mask_length = mast_params.get('pretraining', {}).get('masking', {}).get('length', 8)
            
            # Create a data generator for pretraining
            def pretrain_generator():
                while True:
                    # Select random batch
                    indices = np.random.choice(len(X_pretrain), size=pretrain_batch_size)
                    batch_x = X_pretrain[indices]
                    
                    # Add channel dimension if needed
                    if len(batch_x.shape) == 3:
                        batch_x = np.expand_dims(batch_x, axis=-1)
                    
                    # Apply masking
                    masked_x, _ = apply_masking(
                        batch_x, 
                        mask_probability=mask_probability,
                        mask_length=mask_length
                    )
                    
                    # The model should reconstruct the original unmasked input
                    yield masked_x, batch_x
            
            # Create tf.data.Dataset from generator
            pretrain_dataset = tf.data.Dataset.from_generator(
                pretrain_generator,
                output_signature=(
                    tf.TensorSpec(shape=(None, *target_shape, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, *target_shape, 1), dtype=tf.float32)
                )
            ).prefetch(tf.data.AUTOTUNE)
            
            # Create LR schedule for pretraining
            pretrain_lr_schedule = create_lr_schedule(
                pretrain_lr,
                warmup_epochs=5,
                decay_epochs=pretrain_epochs
            )
            
            # Compile the pretraining model
            pretrain_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=pretrain_lr),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['mse']
            )
            
            # Create callbacks for pretraining
            pretrain_callbacks = [
                tf.keras.callbacks.LearningRateScheduler(pretrain_lr_schedule),
                tf.keras.callbacks.TensorBoard(
                    log_dir=f"logs/log_mast/pretrain_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    histogram_freq=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath="model/MAST/pretrain_model.keras",
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
            
            # Train the pretraining model
            pretrain_model.fit(
                pretrain_dataset,
                steps_per_epoch=len(X_pretrain) // pretrain_batch_size,
                epochs=pretrain_epochs,
                callbacks=pretrain_callbacks,
                verbose=1
            )
            
            logger.info("MAST pretraining completed")
            
            # Save the pretrained weights
            pretrain_model.save_weights("model/MAST/pretrain_weights.keras")
            
            # Load the pretrained weights into the fine-tuning model
            # The shared Transformer layers will have the same names
            logger.info("Transferring pretrained weights to fine-tuning model")
            finetune_model.load_weights("model/MAST/pretrain_weights.keras", by_name=True, skip_mismatch=True)
        
        # Starting fine-tuning
        logger.info("Starting MAST fine-tuning phase for anomaly detection")
        # Optuna hyperparameter optimization
        opt_cfg = config.get('optimization', {}).get('optuna', {})
        if opt_cfg.get('enable', False):
            # Prepare datasets once
            train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels_expanded))\
                .shuffle(buffer_size=train_data.shape[0]).batch(training_params.get('batch_size', 32)).prefetch(tf.data.AUTOTUNE)
            val_ds = tf.data.Dataset.from_tensor_slices((val_data, val_labels))\
                .batch(training_params.get('batch_size', 32)).prefetch(tf.data.AUTOTUNE)
            def objective(trial):
                # Suggest hyperparameters
                # Ensure LR bounds are floats, then sample on log scale
                lr_low = float(opt_cfg['parameters']['learning_rate']['low'])
                lr_high = float(opt_cfg['parameters']['learning_rate']['high'])
                lr = trial.suggest_float('learning_rate', lr_low, lr_high, log=True)
                num_layers = trial.suggest_int('num_encoder_layers',
                    opt_cfg['parameters']['num_encoder_layers']['low'],
                    opt_cfg['parameters']['num_encoder_layers']['high'],
                    step=opt_cfg['parameters']['num_encoder_layers'].get('step',1))
                # Use suggest_float (uniform) for dropout_rate
                drop = trial.suggest_float('dropout_rate',
                    opt_cfg['parameters']['dropout_rate']['low'],
                    opt_cfg['parameters']['dropout_rate']['high'])
                # Use suggest_float (uniform) for mlp_dropout
                mlp_drop = trial.suggest_float('mlp_dropout',
                    opt_cfg['parameters']['mlp_dropout']['low'],
                    opt_cfg['parameters']['mlp_dropout']['high'])
                # Retrieve compile configuration for focal loss parameters
                compile_cfg = config.get('fit', {}).get('compile', {})
                # Update parameters for this trial
                transformer_params['num_encoder_layers'] = num_layers
                transformer_params['dropout_rate'] = drop
                transformer_params['mlp_dropout'] = mlp_drop
                training_params['learning_rate'] = lr
                # Build and compile model
                finetune_model = create_mast_model(target_shape, mast_params, transformer_params)[1]
                optimizer = tf.keras.optimizers.experimental.AdamW(
                    learning_rate=lr, weight_decay=training_params.get('weight_decay',1e-3)
                )
                finetune_model.compile(
                    optimizer=optimizer,
                    loss=focal_loss(gamma=compile_cfg.get('focal_loss',{}).get('gamma',2.0),
                                    alpha=compile_cfg.get('focal_loss',{}).get('alpha',0.25)),
                    metrics=['accuracy']
                )
                # Add early stopping callback
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True
                )

                # Train the model with early stopping
                history = finetune_model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=opt_cfg['trial_epochs'],  # Use trial_epochs for quick evaluation
                    verbose=1,
                    callbacks=[early_stopping]  # Add early stopping here
                )
                return float(history.history['val_accuracy'][-1])
            sampler = getattr(optuna.samplers, opt_cfg.get('sampler','TPESampler'))()
            study = optuna.create_study(direction=opt_cfg.get('direction','maximize'), sampler=sampler)
            study.optimize(objective, n_trials=opt_cfg.get('n_trials',20), timeout=opt_cfg.get('timeout',None))
            best = study.best_params
            logger.info(f"Optuna found best parameters: {best}")
            # Save best parameters to a separate YAML file
            # Ensure optuna_result_file is set and valid
            optuna_result_file = config.get('optimization', {}).get('optuna', {}).get('result_file', 'result/result_MAST/optuna_best_params.yaml')
            if not optuna_result_file:
                raise ValueError("optuna_result_file is not set in the configuration.")

            # Ensure the result directory for Optuna exists
            optuna_result_dir = os.path.dirname(optuna_result_file)
            os.makedirs(optuna_result_dir, exist_ok=True)
            with open(optuna_result_file, 'w') as yaml_file:
                yaml.dump(best, yaml_file, default_flow_style=False)
            logger.info(f"Best hyperparameters saved to {optuna_result_file}")
            # Apply best hyperparameters
            training_params['learning_rate'] = best['learning_rate']
            transformer_params['num_encoder_layers'] = best['num_encoder_layers']
            transformer_params['dropout_rate'] = best['dropout_rate']
            transformer_params['mlp_dropout'] = best['mlp_dropout']
        
        # Create cosine annealing LR schedule and AdamW optimizer with weight decay
        initial_lr = training_params.get('learning_rate', 1e-5)
        decay_steps = training_params.get('decay_steps', 10000)
        fit_cfg = config.get('fit', {})
        compile_cfg = fit_cfg.get('compile', {})
        loss_type = compile_cfg.get('loss', 'binary_crossentropy')
        if loss_type == 'focal_loss':
            fl_cfg = compile_cfg.get('focal_loss', {})
            loss_fn = focal_loss(gamma=fl_cfg.get('gamma', 2.0), alpha=fl_cfg.get('alpha', 0.25))
        else:
            loss_fn = tf.keras.losses.BinaryCrossentropy()
        lr_cfg = fit_cfg.get('lr_scheduler', {})
        if lr_cfg.get('type') == 'cosine_annealing_restarts':
            first_decay = lr_cfg.get('first_decay_steps', decay_steps)
            t_mul = lr_cfg.get('t_mul', 2.0)
            alpha = lr_cfg.get('alpha', 0.0)
            lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=initial_lr,
                first_decay_steps=first_decay,
                t_mul=t_mul,
                m_mul=1.0,
                alpha=alpha
            )
        else:
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=initial_lr, decay_steps=decay_steps
            )
        weight_decay = training_params.get('weight_decay', 1e-3)
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr_schedule, weight_decay=weight_decay
        )
        finetune_model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        # Add channel dimension if needed
        if len(train_data.shape) == 3:
            train_data = np.expand_dims(train_data, axis=-1)
            val_data = np.expand_dims(val_data, axis=-1)
        
        # Build tf.data datasets for efficient training
        batch_size = training_params.get('batch_size', 32)
        train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels_expanded))
        train_ds = train_ds.shuffle(buffer_size=train_data.shape[0]).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        # Setup callbacks for fine-tuning
        callbacks = []
        # Early stopping
        es_cfg = fit_cfg.get('early_stopping', {})
        if es_cfg.get('enabled', False):
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=es_cfg.get('monitor', 'val_loss'),
                    patience=es_cfg.get('patience', 10),
                    min_delta=es_cfg.get('min_delta', 0.0),
                    restore_best_weights=es_cfg.get('restore_best_weights', True)
                )
            )
        # LR scheduler via ReduceLROnPlateau
        lr_cfg = fit_cfg.get('lr_scheduler', {})
        if lr_cfg.get('enabled', False):
            callbacks.append(
                ReduceLROnPlateau(
                    monitor=lr_cfg.get('monitor', 'val_loss'),
                    factor=lr_cfg.get('factor', 0.5),
                    patience=lr_cfg.get('patience', 5),
                    min_delta=lr_cfg.get('min_delta', 0.0),
                    cooldown=lr_cfg.get('cooldown', 0),
                    min_lr=lr_cfg.get('min_lr', 0.0)
                )
            )
        # Model checkpointing
        ckpt_cfg = fit_cfg.get('checkpointing', {})
        if ckpt_cfg.get('enabled', False):
            callbacks.append(
                ModelCheckpoint(
                    filepath=model_path,
                    monitor=ckpt_cfg.get('monitor', 'val_accuracy'),
                    mode=ckpt_cfg.get('mode', 'max'),
                    save_best_only=ckpt_cfg.get('save_best_only', True),
                    save_weights_only=True  # avoid unsupported full model save options
                )
            )

        # Add a custom callback to log predictions and labels during training
        class DebugMetricsCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                precision = logs.get('precision') or logs.get('Precision')
                recall = logs.get('recall') or logs.get('Recall')
                auc = logs.get('auc') or logs.get('AUC')
                print(f"Epoch {epoch + 1}: Precision={precision}, Recall={recall}, AUC={auc}")

        # Add DebugMetricsCallback to the list of callbacks
        callbacks.append(DebugMetricsCallback())

        # Add DebugLossCallback to the training callbacks
        class DebugLossCallback(tf.keras.callbacks.Callback):
            def on_batch_end(self, batch, logs=None):
                logs = logs or {}
                loss = logs.get('loss')
                if loss is not None and (np.isnan(loss) or np.isinf(loss)):
                    logger.warning(f"NaN or Inf loss detected at batch {batch}. Logs: {logs}")

        callbacks.append(DebugLossCallback())

        # Ensure class weights are used if the dataset is imbalanced
        class_weights = training_params.get('class_weights', None)
        if class_weights:
            print(f"Using class weights: {class_weights}")
            history = finetune_model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=training_params.get('epochs', 1), #debug 100
                callbacks=callbacks,
                class_weight=class_weights
            )
        else:
            history = finetune_model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=training_params.get('epochs', 1), #debug 100
                callbacks=callbacks
            )
        
        train_end = time.time()
        model_training_time_seconds = train_end - exec_start

        # Generate and save training loss and accuracy graphs
        viz = Visualizer(param=config)
        viz.loss_plot(history)
        loss_acc_path = os.path.join(result_dir, 'loss_accuracy.png')
        viz.save_figure(loss_acc_path)
        logger.info(f"Saved training curves to {loss_acc_path}")
        
        # Save the final model in SavedModel format or fallback to saving weights
        try:
            model.save(model_path, save_format='tf')  # Use SavedModel format
        except ValueError as e:
            logger.warning(f"Failed to save model in SavedModel format: {e}. Saving weights instead.")
            model.save_weights(model_path + '_weights.h5')
        
        # Save training history
        with open('pickle/pickle_mast/training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)
    
    # Ensure the model is initialized before use
    if 'model' not in locals():
        logger.error("Model is not initialized. Creating a default model.")
        model = create_mast_model(target_shape, mast_params, transformer_params)[1]

    # Evaluate model on test set
    logger.info("Evaluating model on test set")
    
    # Process test data
    test_data, test_labels_expanded = list_to_spectrograms(
        test_files,
        test_labels,
        msg="generate test_dataset",
        augment=False,
        param=config,
        batch_size=20
    )
    
    # Preprocess test data
    test_data = preprocess_spectrograms(test_data, target_shape)
    
    # Apply same normalization to test data as was applied to training data
    if train_std > 1e-6:
        test_data = (test_data - train_mean) / train_std
    else:
        test_data = test_data - train_mean
    
    # Ensure test data has channel dimension
    if len(test_data.shape) == 3:
        test_data = np.expand_dims(test_data, axis=-1)
    
    # Model evaluation
    test_results = model.evaluate(test_data, test_labels_expanded, verbose=1)
    logger.info(f"Test results: {dict(zip(model.metrics_names, test_results))}")
    
    # Generate predictions
    y_pred = model.predict(test_data)
    y_pred_binary = (y_pred > 0.5).astype(int)
    # Build classification report dict
    class_report = classification_report(test_labels_expanded, y_pred_binary, output_dict=True)

    # Plot and save confusion matrix
    viz = Visualizer(param=config)
    viz.plot_confusion_matrix(test_labels_expanded.flatten(), y_pred_binary.flatten())
    cm_path = os.path.join(result_dir, 'confusion_matrix.png')
    viz.save_figure(cm_path)
    logger.info(f"Saved confusion matrix to {cm_path}")

    # Total execution time
    exec_end = time.time()
    execution_time_seconds = exec_end - exec_start
    # Calculate metrics
    accuracy = float(metrics.accuracy_score(test_labels_expanded, y_pred_binary))

    # Extract training and validation accuracy from history
    train_acc = float(history.history.get('accuracy', [0])[-1])
    val_acc = float(history.history.get('val_accuracy', [0])[-1])

    # Extract per-class metrics and support
    f1_0 = float(class_report['0.0']['f1-score'])
    f1_1 = float(class_report['1.0']['f1-score'])
    prec_0 = float(class_report['0.0']['precision'])
    prec_1 = float(class_report['1.0']['precision'])
    rec_0 = float(class_report['0.0']['recall'])
    rec_1 = float(class_report['1.0']['recall'])
    sup_0 = int(class_report['0.0']['support'])
    sup_1 = int(class_report['1.0']['support'])

    # Summary results with requested custom fields only
    results = {
        'execution_time_seconds': float(execution_time_seconds),
        'model_training_time_seconds': float(model_training_time_seconds),
        'overall_model': {
            'F1Score': {'class_0': f1_0, 'class_1': f1_1},
            'Precision': {'class_0': prec_0, 'class_1': prec_1},
            'Recall': {'class_0': rec_0, 'class_1': rec_1},
            'Support': {'class_0': sup_0, 'class_1': sup_1},
            'TestAccuracy': accuracy,
            'TrainAccuracy': train_acc,
            'ValidationAccuracy': val_acc
        }
    }

    # Save as YAML file
    yaml_file_path = os.path.join(result_dir, config.get('result_file', 'result_MAST.yaml'))
    with open(yaml_file_path, 'w') as f:
        yaml.safe_dump(results, f, default_flow_style=False)
    
    logger.info(f"Test results saved to {yaml_file_path}")
    
    return model


if __name__ == "__main__":
    main()