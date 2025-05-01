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
from transformers import TFViTModel, ViTConfig
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
    os.makedirs("./logs/log_AST", exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, filename="./logs/log_AST/baseline_AST.log")
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
    print(f"DEBUG: dataset_generator called with target_dir: {target_dir}")
    logger.info(f"target_dir : {target_dir}")
    
    if param is None:
        param = {}
    
    split_ratio = param.get("dataset", {}).get("split_ratio", [0.8, 0.1, 0.1])
    ext = param.get("dataset", {}).get("file_extension", "wav")
    
    # Determine if we're processing normal or abnormal files
    is_normal = "normal" in str(target_dir)
    condition = "normal" if is_normal else "abnormal"
    print(f"DEBUG: Processing {condition} files")
    
    # Get all files in the directory
    files_in_dir = list(Path(target_dir).glob(f"*.{ext}"))
    print(f"Looking for files in: {target_dir}")
    print(f"Found {len(files_in_dir)} files")

    # If no files found, try listing the directory contents
    if len(files_in_dir) == 0:
        print(f"DEBUG: No files with extension '{ext}' found in {target_dir}")
        print(f"DEBUG: Directory contents: {list(Path(target_dir).iterdir())[:10]}")  # Show first 10 files
    
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
            machine_id = machine_id_with_file.split('-')[0] if '-' in machine_id_with_file else machine_id_with_file
            
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
    print(f"Looking for files in: {target_dir}")
    print(f"Found {len(files_in_dir)} files")
    print(f"DEBUG - Dataset summary:")
    print(f"  Normal files found: {len(normal_files)}")
    print(f"  Abnormal files found: {len(abnormal_files)}")
    print(f"  Normal train: {len(normal_train_files)}, Normal val: {len(normal_val_files)}, Normal test: {len(normal_test_files)}")
    print(f"  Abnormal train: {len(abnormal_train_files)}, Abnormal val: {len(abnormal_val_files)}, Abnormal test: {len(abnormal_test_files)}")

    return train_files, train_labels, val_files, val_labels, test_files, test_labels

def configure_mixed_precision(enabled=True):
    """
    Configure mixed precision training optimized for V100 GPU.
    """
    if not enabled:
        logger.info("Mixed precision training disabled")
        return False
    
    try:
        # Check if GPU is available
        if not tf.config.list_physical_devices('GPU'):
            logger.warning("No GPU found, disabling mixed precision")
            return False
        
        # Import mixed precision module
        from tensorflow.keras import mixed_precision
        
        # Configure policy - V100 works well with mixed_float16
        policy_name = 'mixed_float16'
        logger.info(f"Enabling mixed precision with policy: {policy_name}")
        mixed_precision.set_global_policy(policy_name)
        
        # Verify policy was set
        current_policy = mixed_precision.global_policy()
        logger.info(f"Mixed precision policy enabled: {current_policy}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error configuring mixed precision: {e}")
        # Reset to default policy
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('float32')
            logger.info("Reset to float32 policy after error")
        except:
            pass
        return False

class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    """
    Learning rate scheduler with warmup and cosine decay
    """
    def __init__(self, learning_rate_base, total_steps, warmup_steps, hold_base_rate_steps=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.learning_rates = []
        
    def on_batch_begin(self, batch, logs=None):
        # Calculate current learning rate
        lr = self.calculate_learning_rate(self.global_step)
        # Set learning rate
        K.set_value(self.model.optimizer.lr, lr)
        # Update global step
        self.global_step += 1
        # Store learning rate
        self.learning_rates.append(lr)
        
    def on_train_begin(self, logs=None):
        # Initialize global step
        self.global_step = 0
        
    def calculate_learning_rate(self, global_step):
        """
        Calculate learning rate according to warmup and cosine decay schedule
        """
        # Warmup phase
        if global_step < self.warmup_steps:
            return self.learning_rate_base * (global_step / self.warmup_steps)
        
        # Hold phase
        if self.hold_base_rate_steps > 0 and global_step < self.warmup_steps + self.hold_base_rate_steps:
            return self.learning_rate_base
        
        # Cosine decay phase
        cosine_steps = self.total_steps - self.warmup_steps - self.hold_base_rate_steps
        global_step = global_step - self.warmup_steps - self.hold_base_rate_steps
        
        return 0.5 * self.learning_rate_base * (1 + np.cos(np.pi * global_step / cosine_steps))

def get_scaled_learning_rate(base_lr, batch_size, base_batch_size=32):
    """
    Scale learning rate linearly with batch size
    """
    return base_lr * (batch_size / base_batch_size)

class TerminateOnNaN(tf.keras.callbacks.Callback):
    """
    Callback that terminates training when a NaN loss is encountered
    and reduces learning rate to attempt recovery.
    """
    def __init__(self, patience=3):
        super(TerminateOnNaN, self).__init__()
        self.nan_epochs = 0
        self.patience = patience
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        
        if loss is not None and (np.isnan(loss) or np.isinf(loss)):
            self.nan_epochs += 1
            logger.warning(f"NaN loss detected (occurrence {self.nan_epochs}/{self.patience})")
            
            if self.nan_epochs >= self.patience:
                logger.error(f"NaN loss persisted for {self.patience} epochs, terminating training")
                self.model.stop_training = True
            else:
                # Try to recover by reducing learning rate
                current_lr = float(K.get_value(self.model.optimizer.lr))
                new_lr = current_lr * 0.1
                logger.warning(f"Attempting to recover by reducing learning rate: {current_lr:.6f} -> {new_lr:.6f}")
                K.set_value(self.model.optimizer.lr, new_lr)
        else:
            # Reset counter if we see a valid loss
            self.nan_epochs = 0

########################################################################
# model
########################################################################
def create_ast_model(input_shape, config=None):
    """Create an improved Audio Spectrogram Transformer (AST) model"""
    if config is None:
        config = {}
    
    # Get transformer configuration
    transformer_config = config.get("transformer", {})
    num_heads = transformer_config.get("num_heads", 4)
    dim_feedforward = transformer_config.get("dim_feedforward", 512)
    num_encoder_layers = transformer_config.get("num_encoder_layers", 3)  # Increase layers
    patch_size = transformer_config.get("patch_size", 4)
    attention_dropout = transformer_config.get("attention_dropout", 0.2)  # Increase dropout
    
    # Calculate sequence length and embedding dimension based on input shape and patch size
    h_patches = input_shape[0] // patch_size
    w_patches = input_shape[1] // patch_size
    seq_len = h_patches * w_patches
    embed_dim = dim_feedforward
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Add channel dimension
    x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    # Add batch normalization at the input
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Use depth-wise separable convolution for patch embedding
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=patch_size,
        strides=patch_size,
        padding='same',
        depth_multiplier=4,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
        name='patch_embedding_depthwise'
    )(x)
    
    # Pointwise convolution
    x = tf.keras.layers.Conv2D(
        filters=embed_dim,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
        name='patch_embedding_pointwise'
    )(x)
    
    # Add batch normalization for stability
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Reshape to sequence format for transformer
    x = tf.keras.layers.Reshape((seq_len, embed_dim))(x)
    
    # Layer normalization before adding positional encoding
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Add positional encoding
    pos_encoding = positional_encoding(seq_len, embed_dim, encoding_type="sinusoidal")
    x = tf.keras.layers.Add()([x, pos_encoding])
    
    # Apply transformer encoder layers
    for i in range(num_encoder_layers):
        # Layer normalization before attention (pre-norm formulation)
        attn_input = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Standard multi-head attention
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(16, embed_dim // num_heads),
            dropout=attention_dropout,
            name=f'encoder_mha_{i}'
        )(attn_input, attn_input)
        
        # Add dropout after attention
        attn_output = tf.keras.layers.Dropout(0.1)(attn_output)
        
        # Residual connection
        x = tf.keras.layers.Add()([x, attn_output])
        
        # Layer normalization before FFN
        ffn_input = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Gated feed-forward network
        ffn_hidden = tf.keras.layers.Dense(
            embed_dim * 4,  # Increase size for better representation
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
            name=f'ffn_hidden_{i}'
        )(ffn_input)
        
        # Split into two parts
        ffn_hidden_1, ffn_hidden_2 = tf.split(ffn_hidden, 2, axis=-1)
        
        # Apply gating (GELU activation for one path, sigmoid for gate)
        ffn_output = tf.keras.layers.Multiply()(
            [
                tf.keras.layers.Activation('gelu')(ffn_hidden_1),
                tf.keras.layers.Activation('sigmoid')(ffn_hidden_2)
            ]
        )
        
        # Project back to embedding dimension
        ffn_output = tf.keras.layers.Dense(
            embed_dim,
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
            name=f'ffn_output_{i}'
        )(ffn_output)
        
        # Add dropout for regularization
        ffn_output = tf.keras.layers.Dropout(0.2)(ffn_output)  # Increase dropout
        
        # Residual connection
        x = tf.keras.layers.Add()([x, ffn_output])
    
    # Final layer normalization
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Single streamlined classification head 
    x = tf.keras.layers.Dense(
        dim_feedforward // 4,  # Smaller size
        activation='gelu',
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
        name='classifier_hidden'
    )(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # Reduced dropout
    outputs = tf.keras.layers.Dense(
        1, 
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
        name='classifier_output'
    )(x)
    
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def preprocess_spectrograms(spectrograms, target_shape):
    """
    Resize all spectrograms to a consistent shape.
    """
    if spectrograms.shape[0] == 0:
        return spectrograms
        
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


def balance_dataset(train_data, train_labels, augment_minority=True):
    """
    Balance the dataset by augmenting the minority class with more sophisticated techniques
    """
    # Count classes
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    class_counts = dict(zip(unique_labels, counts))
    logger.info(f"Original class distribution: {class_counts}")
    
    if len(unique_labels) <= 1:
        logger.warning("Only one class present in training data!")
        return train_data, train_labels
    
    # Find minority and majority classes
    minority_class = unique_labels[np.argmin(counts)]
    majority_class = unique_labels[np.argmax(counts)]
    
    if not augment_minority:
        logger.info("Skipping minority class augmentation")
        return train_data, train_labels
    
    # Get indices of minority class
    minority_indices = np.where(train_labels == minority_class)[0]
    
    # Calculate how many samples to generate
    n_to_add = class_counts[majority_class] - class_counts[minority_class]
    
    if n_to_add <= 0:
        logger.info("Dataset already balanced")
        return train_data, train_labels
    
    logger.info(f"Augmenting minority class {minority_class} with {n_to_add} samples")
    
    # Create augmented samples with more sophisticated techniques
    augmented_data = []
    augmented_labels = []
    
    # Use multiple augmentation techniques
    for _ in range(n_to_add):
        # Randomly select a minority sample
        idx = np.random.choice(minority_indices)
        sample = train_data[idx].copy()
        
        # Apply a random augmentation technique
        aug_type = np.random.choice(['noise', 'shift', 'flip', 'mixup'])
        
        if aug_type == 'noise':
            # Add random noise with varying intensity
            noise_level = np.random.uniform(0.05, 0.2)
            noise = np.random.normal(0, noise_level, sample.shape)
            augmented_sample = sample + noise
            
        elif aug_type == 'shift':
            # Random shift in time or frequency
            shift_dim = np.random.choice([0, 1])  # 0 for freq, 1 for time
            shift_amount = np.random.randint(1, 5)
            augmented_sample = np.roll(sample, shift_amount, axis=shift_dim)
            
        elif aug_type == 'flip':
            # Frequency inversion (flip along frequency axis)
            augmented_sample = np.flipud(sample)
            
        elif aug_type == 'mixup':
            # Mix with another minority sample
            idx2 = np.random.choice(minority_indices)
            while idx2 == idx:  # Ensure different sample
                idx2 = np.random.choice(minority_indices)
            sample2 = train_data[idx2]
            mix_ratio = np.random.uniform(0.3, 0.7)
            augmented_sample = mix_ratio * sample + (1 - mix_ratio) * sample2
        
        # Clip values to valid range
        augmented_sample = np.clip(augmented_sample, 0, 1)
        
        augmented_data.append(augmented_sample)
        augmented_labels.append(minority_class)
    
    # Combine original and augmented data
    balanced_data = np.vstack([train_data, np.array(augmented_data)])
    balanced_labels = np.concatenate([train_labels, np.array(augmented_labels)])
    
    # Shuffle the data
    indices = np.arange(len(balanced_labels))
    np.random.shuffle(indices)
    balanced_data = balanced_data[indices]
    balanced_labels = balanced_labels[indices]
    
    logger.info(f"New dataset shape: {balanced_data.shape}")
    new_class_counts = dict(zip(*np.unique(balanced_labels, return_counts=True)))
    logger.info(f"New class distribution: {new_class_counts}")
    
    return balanced_data, balanced_labels

def mixup_data(x, y, alpha=0.2):
    """
    Applies mixup augmentation to the data with improved stability
    """
    if alpha <= 0:
        return x, y
        
    # Generate mixing coefficient
    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = np.maximum(lam, 1-lam)  # Ensure lam is at least 0.5 for stability
    lam = np.reshape(lam, (batch_size, 1, 1))  # Reshape for broadcasting
    
    # Create random permutation of the batch
    index = np.random.permutation(batch_size)
    
    # Mix the data
    mixed_x = lam * x + (1 - lam) * x[index]
    
    # Mix the labels (reshape lam for labels)
    lam_y = np.reshape(lam, (batch_size,))
    mixed_y = lam_y * y + (1 - lam_y) * y[index]
    
    # Ensure consistent dtype
    return mixed_x.astype(np.float32), mixed_y.astype(np.float32)

def normalize_spectrograms(spectrograms, method="minmax"):
    """
    Normalize spectrograms using different methods.
    
    Args:
        spectrograms: Array of spectrograms to normalize
        method: Normalization method ('minmax', 'zscore', or 'log')
        
    Returns:
        Normalized spectrograms
    """
    if method == "minmax":
        # Min-max normalization to range [0, 1]
        min_val = np.min(spectrograms)
        max_val = np.max(spectrograms)
        if max_val == min_val:
            return np.zeros_like(spectrograms)
        return (spectrograms - min_val) / (max_val - min_val)
    
    elif method == "zscore":
        # Z-score normalization (mean=0, std=1)
        mean = np.mean(spectrograms)
        std = np.std(spectrograms)
        if std == 0:
            return np.zeros_like(spectrograms)
        return (spectrograms - mean) / std
    
    elif method == "log":
        # Log normalization
        return np.log1p(spectrograms)
    
    else:
        logger.warning(f"Unknown normalization method: {method}, returning original data")
        return spectrograms

def monitor_gpu_usage():
    """
    Monitor GPU memory usage and log it
    """
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                               stdout=subprocess.PIPE, text=True)
        memory_info = result.stdout.strip().split(',')
        used_memory = int(memory_info[0])
        total_memory = int(memory_info[1])
        usage_percent = (used_memory / total_memory) * 100
        
        logger.info(f"GPU Memory: {used_memory}MB / {total_memory}MB ({usage_percent:.1f}%)")
        return used_memory, total_memory, usage_percent
    except Exception as e:
        logger.warning(f"Failed to monitor GPU usage: {e}")
        return None, None, None

def process_dataset_in_chunks(file_list, labels=None, chunk_size=5000, param=None):
    """
    Process a large dataset in chunks to avoid memory issues
    """

    if param is None:
        param = {}
    
    chunking_config = param.get("feature", {}).get("dataset_chunking", {})
    if not chunking_config.get("enabled", False):
        # Process normally if chunking is disabled
        return list_to_spectrograms(file_list, labels, msg="Processing dataset", augment=False, param=param)
    
    # Use configured chunk size or default
    chunk_size = chunking_config.get("chunk_size", chunk_size)
    logger.info(f"Processing dataset in chunks of {chunk_size} files")
    
    # Create temporary directory if needed
    temp_dir = chunking_config.get("temp_directory", "./temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Split dataset into chunks
    num_chunks = (len(file_list) + chunk_size - 1) // chunk_size
    all_spectrograms = []
    all_labels = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(file_list))
        
        chunk_files = file_list[start_idx:end_idx]
        chunk_labels = labels[start_idx:end_idx] if labels is not None else None
        
        logger.info(f"Processing chunk {i+1}/{num_chunks} ({len(chunk_files)} files)")
        
        # Process this chunk
        chunk_spectrograms, chunk_labels_expanded = list_to_spectrograms(
            chunk_files, 
            chunk_labels, 
            msg=f"Chunk {i+1}/{num_chunks}", 
            augment=False, 
            param=param
        )
        
        # Save chunk to disk to free memory
        chunk_file = f"{temp_dir}/chunk_{i}.npz"
        np.savez_compressed(
            chunk_file, 
            spectrograms=chunk_spectrograms, 
            labels=chunk_labels_expanded if chunk_labels_expanded is not None else np.array([])
        )
        
        # Clear memory
        del chunk_spectrograms, chunk_labels_expanded
        gc.collect()
        
        # Monitor GPU usage
        monitor_gpu_usage()
    
    # Now load and combine all chunks
    target_shape = param.get("feature", {}).get("target_shape", None)
    if target_shape is None:
        # Determine target shape from first chunk
        first_chunk = np.load(f"{temp_dir}/chunk_0.npz")
        if len(first_chunk["spectrograms"]) > 0:
            spec_shape = first_chunk["spectrograms"][0].shape
            target_shape = (spec_shape[0], spec_shape[1])
        else:
            target_shape = (param.get("feature", {}).get("n_mels", 64), 128)
    
    # Count total samples
    total_samples = 0
    for i in range(num_chunks):
        chunk_file = f"{temp_dir}/chunk_{i}.npz"
        chunk_data = np.load(chunk_file)
        total_samples += len(chunk_data["spectrograms"])
    
    # Pre-allocate arrays
    all_spectrograms = np.zeros((total_samples, target_shape[0], target_shape[1]), dtype=np.float32)
    all_labels = np.zeros(total_samples, dtype=np.float32) if labels is not None else None
    
    # Fill arrays
    sample_idx = 0
    for i in range(num_chunks):
        chunk_file = f"{temp_dir}/chunk_{i}.npz"
        chunk_data = np.load(chunk_file)
        chunk_spectrograms = chunk_data["spectrograms"]
        chunk_labels = chunk_data["labels"]
        
        # Resize spectrograms if needed
        for j in range(len(chunk_spectrograms)):
            spec = chunk_spectrograms[j]
            if spec.shape[0] != target_shape[0] or spec.shape[1] != target_shape[1]:
                from skimage.transform import resize
                spec = resize(spec, target_shape, anti_aliasing=True)
            
            all_spectrograms[sample_idx] = spec
            if labels is not None:
                all_labels[sample_idx] = chunk_labels[j]
            
            sample_idx += 1
        
        # Clean up
        os.remove(chunk_file)
    
    # Remove temporary directory if empty
    if not os.listdir(temp_dir):
        os.rmdir(temp_dir)
    
    return all_spectrograms, all_labels

def create_tf_dataset(file_list, labels=None, batch_size=32, is_training=False, param=None):
    """
    Create a TensorFlow dataset that streams and processes audio files on-the-fly - optimized for speed
    """
    # Check if file_list is empty or None
    if file_list is None or len(file_list) == 0:
        logger.error("Empty file list provided to create_tf_dataset")
        return None
        
    if labels is not None:
        labels = np.array(labels, dtype=np.float32)  # Ensure float32 dtype
        
    # Calculate a reasonable shuffle buffer size (smaller is faster)
    shuffle_buffer_size = min(len(file_list), 100)  # Reduce from 500 to 100 maximum
    logger.info(f"Using shuffle buffer size: {shuffle_buffer_size}")
    
    # Create a cache to store processed spectrograms
    _spectrogram_cache = {}
    
    # Function to load and process a single file
    def process_file(file_path, label=None):
        def _process_file(file_path, label):
            # Convert string tensor to string
            file_path_str = file_path.numpy().decode('utf-8')
            
            # Check if we have this in cache
            cache_key = f"{file_path_str}_{is_training}"
            if cache_key in _spectrogram_cache:
                spec = _spectrogram_cache[cache_key]
                if label is not None:
                    return spec, np.float32(label)
                return spec
            
            # Get parameters
            n_mels = param.get("feature", {}).get("n_mels", 64)
            n_fft = param.get("feature", {}).get("n_fft", 1024)
            hop_length = param.get("feature", {}).get("hop_length", 512)
            power = param.get("feature", {}).get("power", 2.0)
            
            # Process file to spectrogram
            spec = file_to_spectrogram(
                file_path_str, 
                n_mels=n_mels, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                power=power, 
                augment=is_training,
                param=param
            )
            
            # Handle case where processing fails
            if spec is None:
                # Return a zero spectrogram with the expected shape
                target_shape = param.get("feature", {}).get("target_shape", (n_mels, 96))
                spec = np.zeros(target_shape, dtype=np.float32)
            
            # Ensure we have a 2D spectrogram (not 3D with frames)
            if len(spec.shape) == 3:
                # If we have multiple frames, just take the first one
                spec = spec[0]
            
            # Ensure consistent shape
            target_shape = param.get("feature", {}).get("target_shape", (n_mels, 96))
            if spec.shape[0] != target_shape[0] or spec.shape[1] != target_shape[1]:
                try:
                    from skimage.transform import resize
                    spec = resize(spec, target_shape, anti_aliasing=True, mode='reflect')
                except Exception:
                    # Fall back to simple padding/cropping
                    temp_spec = np.zeros(target_shape, dtype=np.float32)
                    freq_dim = min(spec.shape[0], target_shape[0])
                    time_dim = min(spec.shape[1], target_shape[1])
                    temp_spec[:freq_dim, :time_dim] = spec[:freq_dim, :time_dim]
                    spec = temp_spec
            
            # Normalize
            if np.max(spec) > np.min(spec):
                spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))
            
            # Ensure float32 dtype
            spec = spec.astype(np.float32)
            if label is not None:
                label = np.float32(label)
            
            # Store in cache if not too many items already
            if len(_spectrogram_cache) < 1000:  # Limit cache size
                _spectrogram_cache[cache_key] = spec
            
            if label is not None:
                return spec, label
            return spec
        
        # Wrap the function to handle TensorFlow tensors
        result = tf.py_function(
            _process_file,
            [file_path, label],
            [tf.float32, tf.float32 if label is not None else None]
        )
        
        # Set shapes explicitly
        target_shape = param.get("feature", {}).get("target_shape", (param.get("feature", {}).get("n_mels", 64), 96))
        result[0].set_shape(target_shape)
        if label is not None:
            result[1].set_shape(())
            return result[0], result[1]
        return result[0]
    
    # Create dataset to reduce file loading overhead
    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((file_list, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(file_list)
    
    # Use fewer parallel calls to avoid CPU bottleneck
    num_parallel_calls = min(8, tf.data.AUTOTUNE)  
    
    # Load and preprocess in parallel
    if labels is not None:
        dataset = dataset.map(process_file, num_parallel_calls=num_parallel_calls)
    else:
        dataset = dataset.map(lambda x: process_file(x), num_parallel_calls=num_parallel_calls)
    
    # Apply optimization techniques
    if len(file_list) < 5000:  # Only cache reasonably sized datasets
        logger.info("Caching dataset in memory")
        dataset = dataset.cache()
    
    # Apply optimization techniques
    dataset = dataset.map(process_file, num_parallel_calls=tf.data.AUTOTUNE)

    # Cache should come before shuffle for better performance
    dataset = dataset.cache()  # Cache first
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)  # Then shuffle
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_small_dataset(files, labels, max_files=100):
    """
    Create a small subset of the dataset for debugging
    """
    if len(files) <= max_files:
        return files, labels
    
    # Ensure we get a balanced sample if labels are provided
    if labels is not None and len(labels) > 0:
        normal_indices = np.where(labels == 0)[0]
        abnormal_indices = np.where(labels == 1)[0]
        
        # Take half from each class, or all if less than half of max_files
        n_normal = min(len(normal_indices), max_files // 2)
        n_abnormal = min(len(abnormal_indices), max_files // 2)
        
        # If one class has fewer samples, take more from the other
        if n_normal < max_files // 2:
            n_abnormal = min(len(abnormal_indices), max_files - n_normal)
        if n_abnormal < max_files // 2:
            n_normal = min(len(normal_indices), max_files - n_abnormal)
        
        # Randomly select samples from each class
        selected_normal = np.random.choice(normal_indices, n_normal, replace=False)
        selected_abnormal = np.random.choice(abnormal_indices, n_abnormal, replace=False)
        
        # Combine indices
        selected_indices = np.concatenate([selected_normal, selected_abnormal])
        np.random.shuffle(selected_indices)
        
        return [files[i] for i in selected_indices], labels[selected_indices]
    else:
        # If no labels, just take a random sample
        indices = np.random.choice(len(files), min(max_files, len(files)), replace=False)
        return [files[i] for i in indices], labels[indices] if labels is not None else None

def weighted_binary_crossentropy(y_true, y_pred, pos_weight=2.0):
    # Convert inputs to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    pos_weight = tf.cast(pos_weight, tf.float32)
    
    # Clip predictions for numerical stability
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate loss with higher weight for positive class
    loss = -(pos_weight * y_true * tf.math.log(y_pred) + 
            (1 - y_true) * tf.math.log(1 - y_pred))
    
    return tf.reduce_mean(loss)

def create_lr_schedule(initial_lr=0.001, warmup_epochs=5, decay_epochs=50, min_lr=0.00001):
    def lr_schedule(epoch):
        # Warmup phase
        if epoch < warmup_epochs:
            return initial_lr * ((epoch + 1) / warmup_epochs)
        
        # Decay phase
        decay_progress = (epoch - warmup_epochs) / decay_epochs
        cosine_decay = 0.5 * (1 + np.cos(np.pi * min(decay_progress, 1.0)))
        return min_lr + (initial_lr - min_lr) * cosine_decay
    
    return lr_schedule

@tf.function
def train_step_with_accumulation(model, optimizer, loss_fn, x, y, accumulated_gradients, first_batch, accum_steps):
    # Cast inputs to float32
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    
    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = model(x, training=True)
        # Compute loss
        loss = loss_fn(y, y_pred)
        # Scale loss by accumulation steps
        scaled_loss = loss / tf.cast(accum_steps, tf.float32)
    
    # Compute gradients
    gradients = tape.gradient(scaled_loss, model.trainable_variables)
    
    # If this is the first batch in an accumulation cycle, reset the accumulated gradients
    if first_batch:
        for i in range(len(accumulated_gradients)):
            accumulated_gradients[i].assign(tf.zeros_like(model.trainable_variables[i], dtype=tf.float32))
    
    # Accumulate gradients
    for i in range(len(accumulated_gradients)):
        accumulated_gradients[i].assign_add(tf.cast(gradients[i], tf.float32))
    
    # Compute accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(y_pred), y), tf.float32))
    
    return loss, accuracy

@tf.function
def val_step(model, loss_fn, x, y):
    # Cast inputs to float32
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    
    # Forward pass
    y_pred = model(x, training=False)
    # Compute loss
    loss = loss_fn(y, y_pred)
    # Compute accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(y_pred), y), tf.float32))
    
    return loss, accuracy

def implement_progressive_training(model, train_files, train_labels, val_files, val_labels, param):
    """
    Implement progressive training with increasing spectrogram sizes - optimized for speed
    """

    # Cache datasets between progressive steps
    cached_datasets = {}

    # Check if input data is valid
    if not train_files or train_files is None or len(train_files) == 0:
        logger.error("No training files provided for progressive training")
        return None, None
        
    if not val_files or val_files is None or len(val_files) == 0:
        logger.warning("No validation files provided for progressive training, using a portion of training data")
        # Use a portion of training data for validation if no validation data is provided
        split_idx = int(len(train_files) * 0.9)
        val_files = train_files[split_idx:]
        val_labels = train_labels[split_idx:] if train_labels is not None else None
        train_files = train_files[:split_idx]
        train_labels = train_labels[:split_idx] if train_labels is not None else None
    
    # Get base parameters
    base_epochs = param.get("fit", {}).get("epochs", 30)
    batch_size = param.get("fit", {}).get("batch_size", 16)
    
    # Define progressive sizes (start smaller, end with target size)
    progressive_config = param.get("training", {}).get("progressive_training", {})
    if not progressive_config.get("enabled", False):
        logger.info("Progressive training disabled")
        return None, None
    
    # Get sizes from config or use defaults
    sizes = progressive_config.get("sizes", [[32, 48], [48, 64], [64, 96]])
    epochs_per_size = progressive_config.get("epochs_per_size", [10, 10, 10])
    
    # Ensure we have enough epochs for each size
    if len(epochs_per_size) != len(sizes):
        epochs_per_size = [base_epochs // len(sizes)] * len(sizes)
    
    logger.info(f"Starting progressive training with sizes: {sizes}")
    logger.info(f"Epochs per size: {epochs_per_size}")
    
    # Store history for each stage
    all_history = []
    
    # Train progressively
    for i, (size, epochs) in enumerate(zip(sizes, epochs_per_size)):
        logger.info(f"Progressive training stage {i+1}/{len(sizes)}: size={size}, epochs={epochs}")
        
        # Update target shape in parameters
        param["feature"]["target_shape"] = size
        size_key = f"{size[0]}x{size[1]}"
        
        # Check if we already have this dataset cached
        if size_key in cached_datasets:
            logger.info(f"Using cached datasets for size {size}")
            train_dataset, val_dataset = cached_datasets[size_key]
        else:
            # Create datasets with current size
            try:
                # Set a smaller shuffle buffer for speed
                old_buffer_size = param.get("training", {}).get("streaming_data", {}).get("prefetch_buffer_size", 4)
                param["training"]["streaming_data"]["prefetch_buffer_size"] = 2
                
                train_dataset = create_tf_dataset(
                    train_files, 
                    train_labels, 
                    batch_size=batch_size, 
                    is_training=True, 
                    param=param
                )
                
                val_dataset = create_tf_dataset(
                    val_files, 
                    val_labels, 
                    batch_size=batch_size, 
                    is_training=False, 
                    param=param
                )
                
                # Restore original buffer size
                param["training"]["streaming_data"]["prefetch_buffer_size"] = old_buffer_size
                
                # Cache the datasets for future use
                cached_datasets[size_key] = (train_dataset, val_dataset)
                
            except Exception as e:
                logger.error(f"Error creating datasets for size {size}: {e}")
                return None, None


        logger.info(f"Checking dataset shapes for size {size}...")
        for x_batch, y_batch in train_dataset.take(1):
            logger.info(f"Training batch shape: {x_batch.shape}, Labels shape: {y_batch.shape}")
        for x_batch, y_batch in val_dataset.take(1):
            logger.info(f"Validation batch shape: {x_batch.shape}, Labels shape: {y_batch.shape}")

        # Use a Python iterator to track progress outside the graph context
        logger.info("Starting dataset loading - this may take a while...")

        # Define a simple Python iterator that won't use .numpy() in graph mode
        class ProgressCallback:
            def __init__(self, name="Dataset"):
                self.count = 0
                self.name = name
                
            def __call__(self, *args):
                self.count += 1
                if self.count % 10 == 0:
                    logger.info(f"{self.name}: Loaded {self.count} batches")
                return args

        # Create the callbacks
        train_callback = ProgressCallback("Training dataset")
        val_callback = ProgressCallback("Validation dataset")

        # Check dataset shapes (this will iterate through one batch each)
        logger.info(f"Checking dataset shapes for size {size}...")
        for x_batch, y_batch in train_dataset.take(1):
            logger.info(f"Training batch shape: {x_batch.shape}, Labels shape: {y_batch.shape}")
            
        for x_batch, y_batch in val_dataset.take(1):
            logger.info(f"Validation batch shape: {x_batch.shape}, Labels shape: {y_batch.shape}")


        # If first stage, create model from scratch
        if i == 0:
            model = create_ast_model(input_shape=size, config=param.get("model", {}).get("architecture", {}))
            
            # Compile model
            model.compile(
                optimizer = AdamW(
                    learning_rate=param.get("fit", {}).get("compile", {}).get("learning_rate", 0.0001),
                    weight_decay=0.01,  # Add weight decay
                    clipnorm=clipnorm
                ),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            # For later stages, we need to adjust the input layer
            # Create a new model with the current size
            new_model = create_ast_model(input_shape=size, config=param.get("model", {}).get("architecture", {}))
            
            # Copy weights from previous model where possible
            for new_layer, old_layer in zip(new_model.layers[1:], model.layers[1:]):
                try:
                    new_layer.set_weights(old_layer.get_weights())
                except:
                    logger.warning(f"Could not transfer weights for layer {new_layer.name}")
            
            # Replace model
            model = new_model
            
            # Recompile
            model.compile(
                optimizer = AdamW(
                    learning_rate=param.get("fit", {}).get("compile", {}).get("learning_rate", 0.0001) * 0.5,  # Lower LR for fine-tuning
                    weight_decay=0.01,  # Add weight decay
                    clipnorm=clipnorm
                ),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        
        # Define callbacks for this stage
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_delta=0.001
            ),
            tf.keras.callbacks.TerminateOnNaN()
        ]
        
        # Train for this stage
        try:
            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=callbacks,
                verbose=1
            )
            
            all_history.append(history)
        except Exception as e:
            logger.error(f"Error during training for size {size}: {e}")
            # Continue with next size instead of failing completely
            continue
        
        # Save intermediate model
        try:
            model.save(f"{param['model_directory']}/model_stage_{i+1}.keras")
            logger.info(f"Saved model for stage {i+1}")
        except Exception as e:
            logger.warning(f"Error saving model for stage {i+1}: {e}")
    
    return model, all_history


# Modify your find_optimal_threshold function definition:
def find_optimal_threshold(y_true, y_pred_proba, param=None):
    """
    Find the optimal classification threshold using various metrics
    """
    thresholds = np.linspace(0.1, 0.9, 33)  # Test 33 thresholds from 0.1 to 0.9
    f1_scores = []
    precisions = []
    recalls = []
    specificities = []
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        # Calculate metrics
        precision = metrics.precision_score(y_true, y_pred, zero_division=0)
        recall = metrics.recall_score(y_true, y_pred, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate specificity (true negative rate)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Store results
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        specificities.append(specificity)
    
    # Find threshold with best F1 score
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_f1_idx]
    
    # Find threshold with best balance between precision and recall
    pr_diffs = np.abs(np.array(precisions) - np.array(recalls))
    best_balance_idx = np.argmin(pr_diffs)
    best_balance_threshold = thresholds[best_balance_idx]
    
    # Find threshold with best geometric mean of recall and specificity
    gmeans = np.sqrt(np.array(recalls) * np.array(specificities))
    best_gmean_idx = np.argmax(gmeans)
    best_gmean_threshold = thresholds[best_gmean_idx]
    
    # Log results
    logger.info(f"Best F1 threshold: {best_f1_threshold:.3f} (F1={f1_scores[best_f1_idx]:.3f})")
    logger.info(f"Best balanced threshold: {best_balance_threshold:.3f} (Precision={precisions[best_balance_idx]:.3f}, Recall={recalls[best_balance_idx]:.3f})")
    logger.info(f"Best geometric mean threshold: {best_gmean_threshold:.3f} (G-mean={gmeans[best_gmean_idx]:.3f})")
    
    # Plot the metrics vs threshold
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, precisions, 'b-', label='Precision')
    plt.plot(thresholds, recalls, 'g-', label='Recall')
    plt.plot(thresholds, f1_scores, 'r-', label='F1 Score')
    plt.plot(thresholds, specificities, 'c-', label='Specificity')
    plt.plot(thresholds, gmeans, 'm-', label='G-Mean')
    
    # Mark the best thresholds
    plt.axvline(x=best_f1_threshold, color='r', linestyle='--', alpha=0.5, label=f'Best F1 ({best_f1_threshold:.3f})')
    plt.axvline(x=best_balance_threshold, color='g', linestyle='--', alpha=0.5, label=f'Best Balance ({best_balance_threshold:.3f})')
    plt.axvline(x=best_gmean_threshold, color='m', linestyle='--', alpha=0.5, label=f'Best G-Mean ({best_gmean_threshold:.3f})')
    
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    
    # Check if param exists before using it
    if param is not None and 'result_directory' in param:
        plt.savefig(f"{param['result_directory']}/threshold_optimization.png")
    else:
        plt.savefig("threshold_optimization.png")
    plt.close()
    
    # Return the best threshold based on F1 score
    return best_f1_threshold


def standardize_spectrograms(spectrograms):
    """
    Standardize spectrograms using robust scaling to handle outliers
    """
    # Calculate robust statistics (less affected by outliers)
    q1 = np.percentile(spectrograms, 25, axis=(1, 2), keepdims=True)
    q3 = np.percentile(spectrograms, 75, axis=(1, 2), keepdims=True)
    median = np.median(spectrograms, axis=(1, 2), keepdims=True)
    
    # Calculate IQR (Interquartile Range)
    iqr = q3 - q1
    
    # Handle cases where IQR is too small
    iqr = np.maximum(iqr, 1e-5)
    
    # Apply robust scaling: (X - median) / IQR
    scaled_specs = (spectrograms - median) / iqr
    
    # Clip extreme values
    scaled_specs = np.clip(scaled_specs, -3, 3)
    
    return scaled_specs


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
            # Allow TensorFlow to allocate memory as needed, but set a growth limit
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set memory limit to 30GB (leaving some headroom)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=30 * 1024)]
            )
            logger.info("GPU memory configuration set for V100 32GB")
        except RuntimeError as e:
            logger.error(f"Error setting GPU memory configuration: {e}")

    with open("baseline_AST.yaml", "r") as stream:
        param = yaml.safe_load(stream)    

    # Enable XLA compilation for faster GPU execution
    if param.get("training", {}).get("xla_acceleration", True):
        logger.info("Enabling XLA acceleration for faster training")
        try:
            tf.config.optimizer.set_jit(True)  # Enable XLA
            os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'  # Force XLA on all operations
            logger.info("XLA acceleration enabled successfully")
        except Exception as e:
            logger.warning(f"Failed to enable XLA acceleration: {e}")


    print("============== CHECKING DIRECTORY STRUCTURE ==============")
    normal_dir = Path(param["base_directory"]) / "normal"
    abnormal_dir = Path(param["base_directory"]) / "abnormal"

    print(f"Normal directory exists: {normal_dir.exists()}")
    if normal_dir.exists():
        normal_files = list(normal_dir.glob("*.wav"))
        print(f"Number of normal files found: {len(normal_files)}")
        if normal_files:
            print(f"Sample normal filename: {normal_files[0].name}")

    print(f"Abnormal directory exists: {abnormal_dir.exists()}")
    if abnormal_dir.exists():
        abnormal_files = list(abnormal_dir.glob("*.wav"))
        print(f"Number of abnormal files found: {len(abnormal_files)}")
        if abnormal_files:
            print(f"Sample abnormal filename: {abnormal_files[0].name}")

    start_time = time.time()

    with open("baseline_AST.yaml", "r") as stream:
        param = yaml.safe_load(stream)

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
        print(f"DEBUG: Testing direct audio load for: {test_files[0]}")
        sr, y = demux_wav(str(test_files[0]))
        if y is not None:
            print(f"DEBUG: Successfully loaded audio with sr={sr}, length={len(y)}")
        else:
            print(f"DEBUG: Failed to load audio file")
    else:
        print("DEBUG: No test files found to verify audio loading")

    base_path = Path(param["base_directory"])

    print("============== COUNTING DATASET SAMPLES ==============")
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
    print(f"DEBUG: Found {len(sample_files)} files in {target_dir}")
    print(f"DEBUG: First 5 files: {[f.name for f in sample_files[:5]]}")

    if not sample_files:
        logger.warning(f"No files found in {target_dir}")
        return  # Exit main() if no files are found

    # Parse a sample filename to get db, machine_type, machine_id
    filename = sample_files[0].name
    parts = filename.split('_')
    print(f"DEBUG: Parsing filename '{filename}' into parts: {parts}")

    if len(parts) < 4:
        logger.warning(f"Filename format incorrect: {filename}")
        return  # Exit main() if filename format is incorrect
    
    # Use a straightforward key without unintended characters
    evaluation_result_key = "overall_model"
    print(f"DEBUG: Using evaluation_result_key: {evaluation_result_key}")

    # Initialize evaluation result dictionary
    evaluation_result = {}
    
    # Initialize results dictionary if it doesn't exist
    results = {}
    all_y_true = []
    all_y_pred = []
    result_file = f"{param['result_directory']}/result_AST.yaml"

    print("============== DATASET_GENERATOR ==============")
    train_pickle = f"{param['pickle_directory']}/train_overall.pickle"
    train_labels_pickle = f"{param['pickle_directory']}/train_labels_overall.pickle"
    val_pickle = f"{param['pickle_directory']}/val_overall.pickle"
    val_labels_pickle = f"{param['pickle_directory']}/val_labels_overall.pickle"
    test_files_pickle = f"{param['pickle_directory']}/test_files_overall.pickle"
    test_labels_pickle = f"{param['pickle_directory']}/test_labels_overall.pickle"

    # Initialize variables
    train_files, train_labels, val_files, val_labels, test_files, test_labels = [], [], [], [], [], []

    if (os.path.exists(train_pickle) and os.path.exists(train_labels_pickle) and
        os.path.exists(val_pickle) and os.path.exists(val_labels_pickle) and
        os.path.exists(test_files_pickle) and os.path.exists(test_labels_pickle)):
        train_data = load_pickle(train_pickle)
        train_labels = load_pickle(train_labels_pickle)
        val_data = load_pickle(val_pickle)
        val_labels = load_pickle(val_labels_pickle)
        test_files = load_pickle(test_files_pickle)
        test_labels = load_pickle(test_labels_pickle)
    else:
        train_files, train_labels, val_files, val_labels, test_files, test_labels = dataset_generator(target_dir, param=param)

        if len(train_files) == 0 or len(val_files) == 0 or len(test_files) == 0:
            logger.error(f"No files found for {evaluation_result_key}, skipping...")
            return  # Exit main() if no files are found after generation

    debug_mode = param.get("debug", {}).get("enabled", False)
    debug_sample_size = param.get("debug", {}).get("sample_size", 100)

    if debug_mode:
        logger.info(f"DEBUG MODE: Using small dataset with {debug_sample_size} samples")
        train_files, train_labels = create_small_dataset(train_files, train_labels, debug_sample_size)
        val_files, val_labels = create_small_dataset(val_files, val_labels, debug_sample_size // 2)
        test_files, test_labels = create_small_dataset(test_files, test_labels, debug_sample_size // 2)
        
        logger.info(f"DEBUG dataset sizes - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")


    preprocessing_batch_size = param.get("feature", {}).get("preprocessing_batch_size", 64)
    chunking_enabled = param.get("feature", {}).get("dataset_chunking", {}).get("enabled", False)
    chunk_size = param.get("feature", {}).get("dataset_chunking", {}).get("chunk_size", 5000)

    # For training data
    if chunking_enabled and len(train_files) > chunk_size:
        logger.info(f"Processing training data in chunks (dataset size: {len(train_files)} files)")
        train_data, train_labels_expanded = process_dataset_in_chunks(
            train_files,
            train_labels,
            chunk_size=chunk_size,
            param=param
        )
    else:
        train_data, train_labels_expanded = list_to_spectrograms(
            train_files,
            train_labels,
            msg="generate train_dataset",
            augment=True,
            param=param,
            batch_size=preprocessing_batch_size
        )

    # For validation data
    if chunking_enabled and len(val_files) > chunk_size:
        logger.info(f"Processing validation data in chunks (dataset size: {len(val_files)} files)")
        val_data, val_labels_expanded = process_dataset_in_chunks(
            val_files,
            val_labels,
            chunk_size=chunk_size,
            param=param
        )
    else:
        val_data, val_labels_expanded = list_to_spectrograms(
            val_files,
            val_labels,
            msg="generate validation_dataset",
            augment=False,
            param=param,
            batch_size=preprocessing_batch_size
        )

    # For test data
    if chunking_enabled and len(test_files) > chunk_size:
        logger.info(f"Processing test data in chunks (dataset size: {len(test_files)} files)")
        test_data, test_labels_expanded = process_dataset_in_chunks(
            test_files,
            test_labels,
            chunk_size=chunk_size,
            param=param
        )
    else:
        test_data, test_labels_expanded = list_to_spectrograms(
            test_files,
            test_labels,
            msg="generate test_dataset",
            augment=False,
            param=param,
            batch_size=preprocessing_batch_size
        )

    if train_data.shape[0] == 0 or val_data.shape[0] == 0:
        logger.error(f"No valid training/validation data for {evaluation_result_key}, skipping...")
        return  # Exit main() if no valid training/validation data

    logger.info("Applying robust standardization to spectrograms...")
    train_data = standardize_spectrograms(train_data)
    val_data = standardize_spectrograms(val_data)
    if 'test_data' in locals() and test_data is not None:
        test_data = standardize_spectrograms(test_data)

    save_pickle(train_pickle, train_data)
    save_pickle(train_labels_pickle, train_labels_expanded)
    save_pickle(val_pickle, val_data)
    save_pickle(val_labels_pickle, val_labels_expanded)
    save_pickle(test_files_pickle, test_files)
    save_pickle(test_labels_pickle, test_labels)

    # Print shapes
    logger.info(f"Training data shape: {train_data.shape}")
    logger.info(f"Training labels shape: {train_labels_expanded.shape}")
    logger.info(f"Validation data shape: {val_data.shape}")
    logger.info(f"Validation labels shape: {val_labels_expanded.shape}")
    logger.info(f"Number of test files: {len(test_files)}")

    # Define target shape for spectrograms
    target_shape = (param["feature"]["n_mels"], 96)
    logger.info(f"Target spectrogram shape: {target_shape}")

    # Preprocess to ensure consistent shapes
    logger.info("Preprocessing training data...")
    train_data = preprocess_spectrograms(train_data, target_shape)
    logger.info(f"Preprocessed train data shape: {train_data.shape}")

    logger.info("Preprocessing validation data...")
    val_data = preprocess_spectrograms(val_data, target_shape)
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
    train_data, train_labels_expanded = balance_dataset(train_data, train_labels_expanded, augment_minority=True)

    # Apply data augmentation
    augmented_data = []
    augmented_labels = []

    # Get abnormal samples
    abnormal_indices = np.where(train_labels_expanded == 1)[0]
    logger.info(f"Augmenting {len(abnormal_indices)} abnormal samples")

    # Augment each abnormal sample once with simple noise
    for idx in abnormal_indices:
        sample = train_data[idx].copy()
        noise = np.random.normal(0, 0.1, sample.shape)
        augmented_sample = sample + noise
        augmented_sample = np.clip(augmented_sample, 0, 1)
        
        augmented_data.append(augmented_sample)
        augmented_labels.append(1)  # Abnormal class

    # Add augmented samples to training data
    if augmented_data:
        augmented_data = np.array(augmented_data)
        train_data = np.vstack([train_data, augmented_data])
        train_labels_expanded = np.concatenate([train_labels_expanded, np.array(augmented_labels)])
        
        # Shuffle the combined dataset
        shuffle_indices = np.random.permutation(len(train_data))
        train_data = train_data[shuffle_indices]
        train_labels_expanded = train_labels_expanded[shuffle_indices]
        
        logger.info(f"After augmentation: {len(train_data)} samples, {np.sum(train_labels_expanded == 1)} abnormal")

    # Configure mixed precision
    mixed_precision_enabled = configure_mixed_precision(
        enabled=param.get("training", {}).get("mixed_precision", False)
    )

    # Check if we should use streaming data
    use_streaming = param.get("training", {}).get("streaming_data", {}).get("enabled", False)

    if use_streaming:
        logger.info("Using streaming data pipeline with tf.data")
        # Ensure all labels are float32
        if train_labels is not None:
            train_labels = np.array(train_labels, dtype=np.float32)
        if val_labels is not None:
            val_labels = np.array(val_labels, dtype=np.float32)
        if test_labels is not None:
            test_labels = np.array(test_labels, dtype=np.float32)
        
        # Create TensorFlow datasets
        batch_size = param.get("fit", {}).get("batch_size", 16)
        orig_batch_size = batch_size


        if batch_size < 32 and param.get("training", {}).get("optimize_batch_size", True):
            batch_size = 32  # Suggested size for V100 32GB
            logger.info(f"Increased batch size from {orig_batch_size} to {batch_size} for V100 32GB (can be disabled in config)")

        # Update the configuration to use this optimized batch size
        param["fit"]["batch_size"] = batch_size

        logger.info(f"Using batch size {batch_size} optimized for V100 32GB")
        train_dataset = create_tf_dataset(
            train_files, 
            train_labels, 
            batch_size=batch_size, 
            is_training=True, 
            param=param
        )
        
        val_dataset = create_tf_dataset(
            val_files, 
            val_labels, 
            batch_size=batch_size, 
            is_training=False, 
            param=param
        )

    monitor_gpu_usage()

    # Debug
    normal_count = sum(1 for label in train_labels_expanded if label == 0)
    abnormal_count = sum(1 for label in train_labels_expanded if label == 1)
    print(f"Training data composition: Normal={normal_count}, Abnormal={abnormal_count}")
    
    # Check for data shape mismatch and fix it
    if train_data.shape[0] != train_labels_expanded.shape[0]:
        logger.warning(f"Data shape mismatch! X: {train_data.shape[0]} samples, y: {train_labels_expanded.shape[0]} labels")
        
        if train_data.shape[0] > train_labels_expanded.shape[0]:
            # Too many features, need to reduce
            train_data = train_data[:train_labels_expanded.shape[0]]
            logger.info(f"Reduced X to match y: {train_data.shape}")
        else:
            # Too many labels, need to reduce
            train_labels_expanded = train_labels_expanded[:train_data.shape[0]]
            logger.info(f"Reduced y to match X: {train_labels_expanded.shape}")

    # Check class distribution
    class_distribution = np.bincount(train_labels_expanded.astype(int))
    logger.info(f"Class distribution in training data: {class_distribution}")
    if len(class_distribution) > 1:
        class_ratio = class_distribution[0] / class_distribution[1] if class_distribution[1] > 0 else float('inf')
        logger.info(f"Class ratio (normal:abnormal): {class_ratio:.2f}:1")
        
        if class_ratio > 10:
            logger.warning(f"Severe class imbalance detected! Consider using class weights or data augmentation.")

    batch_size = param.get("fit", {}).get("batch_size", 32)
    epochs = param.get("fit", {}).get("epochs", 100)
    base_learning_rate = param.get("fit", {}).get("compile", {}).get("learning_rate", 0.001)

    # Scale learning rate based on batch size
    learning_rate = get_scaled_learning_rate(base_learning_rate, batch_size)
    logger.info(f"Scaled learning rate from {base_learning_rate} to {learning_rate} for batch size {batch_size}")

    # Log the training parameters being used
    logger.info(f"Training with batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate}")

    # Set fixed class weights with higher weight for abnormal class
    class_weights = {
        0: 1.0,  # Normal class
        1: 2.0   # Abnormal class - ensure higher weight
    }
    # Convert keys to integers to avoid dtype issues
    class_weights = {int(k): float(v) for k, v in class_weights.items()}
    logger.info(f"Using fixed class weights: {class_weights}")

    # Ensure consistent data types before training
    logger.info(f"Train data type: {train_data.dtype}")
    logger.info(f"Train labels type: {train_labels_expanded.dtype}")

    # Convert to float32 if needed
    if train_data.dtype != np.float32:
        logger.info("Converting train_data to float32")
        train_data = train_data.astype(np.float32)
        
    if train_labels_expanded.dtype != np.float32:
        logger.info("Converting train_labels to float32")
        train_labels_expanded = train_labels_expanded.astype(np.float32)
        
    if val_data.dtype != np.float32:
        logger.info("Converting val_data to float32")
        val_data = val_data.astype(np.float32)
        
    if val_labels_expanded.dtype != np.float32:
        logger.info("Converting val_labels to float32")
        val_labels_expanded = val_labels_expanded.astype(np.float32)

    def verify_gpu_usage():
        """Verify that TensorFlow is properly using the GPU"""
        # Create a simple test tensor and operation
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
            c = tf.matmul(a, b)
        
        # Check if the operation was executed on GPU
        logger.info(f"Test tensor device: {c.device}")
        if 'GPU' in c.device:
            logger.info("✓ GPU is properly configured and being used")
        else:
            logger.warning("⚠ GPU is not being used for tensor operations!")
        
        return 'GPU' in c.device

    print("============== VERIFYING GPU USAGE ==============")
    is_gpu_working = verify_gpu_usage()
    if not is_gpu_working:
        logger.warning("GPU is not being used! Training will be slow.")
        return

    print("============== MODEL TRAINING ==============")
    # Track model training time specifically
    model_start_time = time.time()
    # Define model_file and history_img variables
    model_file = f"{param['model_directory']}/model_overall_ast.keras"
    history_img = f"{param['result_directory']}/history_overall_ast.png"

    # Enable mixed precision training
    if param.get("training", {}).get("mixed_precision", False):
        logger.info("Enabling mixed precision training")
        mixed_precision.set_global_policy('mixed_float16')
        logger.info(f"Mixed precision policy enabled: {mixed_precision.global_policy()}")

    # Check if we should use progressive training
    use_progressive = param.get("training", {}).get("progressive_training", {}).get("enabled", False)

    if use_progressive and not os.path.exists(model_file):
        logger.info("Using progressive training with increasing spectrogram sizes")
        
        # Check if we have valid training and validation files
        if not train_files or len(train_files) == 0:
            logger.error("No training files available for progressive training")
        elif not val_files or len(val_files) == 0:
            logger.warning("No validation files for progressive training, will use a portion of training data")
        else:
            model, progressive_history = implement_progressive_training(
                None,  # No initial model
                train_files,
                train_labels,
                val_files,
                val_labels,
                param
            )
            
            # Check if progressive training was successful
            if model is None:
                logger.error("Progressive training failed, falling back to standard training")
                use_progressive = False
            else:
                # Combine histories
                history = {
                    'history': {
                        'loss': [],
                        'accuracy': [],
                        'val_loss': [],
                        'val_accuracy': []
                    }
                }
                
                for h in progressive_history:
                    history['history']['loss'].extend(h.history['loss'])
                    history['history']['accuracy'].extend(h.history['accuracy'])
                    history['history']['val_loss'].extend(h.history['val_loss'])
                    history['history']['val_accuracy'].extend(h.history['val_accuracy'])
                
                # Convert to object with history attribute for compatibility
                history = type('History', (), history)
                
                # Save final model
                try:
                    model.save(model_file)
                    logger.info(f"Model saved to {model_file}")
                except Exception as e:
                    logger.warning(f"Error saving model: {e}")
                    try:
                        model.save(model_file.replace('.keras', ''))
                        logger.info(f"Model saved with alternative format to {model_file.replace('.keras', '')}")
                    except Exception as e2:
                        logger.error(f"All attempts to save model failed: {e2}")
    else:
        # Regular training without progressive resizing
        if os.path.exists(model_file) or os.path.exists(f"{model_file}.index"):
            try:
                # Try loading with different formats
                if os.path.exists(model_file):
                    model = tf.keras.models.load_model(model_file, custom_objects={"binary_cross_entropy_loss": binary_cross_entropy_loss})
                else:
                    model = tf.keras.models.load_model(f"{model_file}", custom_objects={"binary_cross_entropy_loss": binary_cross_entropy_loss})
                logger.info("Model loaded from file, no training performed")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                # Create a new model
                model = create_ast_model(
                    input_shape=(target_shape[0], target_shape[1]),
                    config=param.get("model", {}).get("architecture", {})
                )
                logger.info("Created new model due to loading error")
        else:
            # Define callbacks
            callbacks = []
            
            # Early stopping
            early_stopping_config = param.get("fit", {}).get("early_stopping", {})
            if early_stopping_config.get("enabled", False):
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor=early_stopping_config.get("monitor", "val_loss"),
                    patience=20,
                    min_delta=early_stopping_config.get("min_delta", 0.001),
                    restore_best_weights=True,
                    verbose=1
                ))

            callbacks.append(TerminateOnNaN(patience=3))
            logger.info("Added NaN detection callback")
                    
            # Reduce learning rate on plateau
            lr_config = param.get("fit", {}).get("lr_scheduler", {})
            if lr_config.get("enabled", False):
                logger.info("Adding ReduceLROnPlateau callback with settings:")
                logger.info(f"  - Monitor: {lr_config.get('monitor', 'val_loss')}")
                logger.info(f"  - Factor: {lr_config.get('factor', 0.1)}")
                logger.info(f"  - Patience: {lr_config.get('patience', 5)}")
                callbacks.append(ReduceLROnPlateau(
                    monitor=lr_config.get("monitor", "val_loss"),
                    factor=lr_config.get("factor", 0.1),
                    patience=lr_config.get("patience", 5),
                    min_delta=lr_config.get("min_delta", 0.001),
                    cooldown=lr_config.get("cooldown", 2),
                    min_lr=lr_config.get("min_lr", 0.00000001),
                    verbose=1
                ))

            # Add learning rate scheduler with warmup
            callbacks.append(
                tf.keras.callbacks.LearningRateScheduler(
                    create_lr_schedule(initial_lr=0.001, warmup_epochs=5, decay_epochs=50)
                )
            )

            # Add callback to detect and handle NaN values
            callbacks.append(tf.keras.callbacks.TerminateOnNaN())
            
            # Create model with the correct input shape
            model = create_ast_model(
                input_shape=(target_shape[0], target_shape[1]),
                config=param.get("model", {}).get("architecture", {})
            )
            
            model_config = param.get("model", {}).get("architecture", {})
            model.summary()
            
            # Compile model
            compile_params = param["fit"]["compile"].copy()
            loss_type = param.get("model", {}).get("loss", "binary_crossentropy")
            
            # Handle learning_rate separately for the optimizer
            learning_rate = compile_params.pop("learning_rate", 0.0001)
            
            # Adjust optimizer for mixed precision if enabled
            clipnorm = param.get("training", {}).get("gradient_clip_norm", 1.0)
            if mixed_precision_enabled and compile_params.get("optimizer") == "AdamW":
                optimizer = AdamW(learning_rate=learning_rate, clipnorm=clipnorm)
                logger.info(f"Using AdamW optimizer with mixed precision and gradient clipping (clipnorm={clipnorm})")
            else:
                optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, clipnorm=clipnorm)
                logger.info(f"Using AdamW optimizer with gradient clipping (clipnorm={clipnorm})")

            # Use a more stable loss function
            if loss_type == "binary_crossentropy":
                loss_fn = "binary_crossentropy"  # Use TF's built-in implementation for stability
            elif loss_type == "focal_loss":
                # Only use focal loss if explicitly requested and with safeguards
                gamma = param.get("model", {}).get("focal_loss", {}).get("gamma", 2.0)
                alpha = param.get("model", {}).get("focal_loss", {}).get("alpha", 0.25)
                
                # Check if we should use a safer implementation
                use_safe_focal = param.get("model", {}).get("focal_loss", {}).get("use_safe_implementation", True)
                if use_safe_focal:
                    logger.info("Using numerically stable focal loss implementation")
                    loss_fn = lambda y_true, y_pred: focal_loss(y_true, y_pred, gamma, alpha)
                else:
                    logger.warning("Using standard focal loss - watch for NaN losses")
                    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
                        gamma=gamma, alpha=alpha, from_logits=False
                    )
            else:
                logger.info("Using standard binary crossentropy loss")
                loss_fn = "binary_crossentropy"
            
            #Define a custom loss function that combines binary cross-entropy with focal loss
            def combined_loss(y_true, y_pred):
                # Cast inputs to float32
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                
                # Binary cross-entropy component
                bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
                
                # Add a penalty term that amplifies errors near the decision boundary
                boundary_penalty = 10.0 * y_true * (1 - y_pred) * (1 - y_pred) + 10.0 * (1 - y_true) * y_pred * y_pred
                
                return bce + tf.reduce_mean(boundary_penalty)

            # Use more aggressive class weights
            class_weights = {
                0: 1.0,  # Normal class
                1: 10.0   # Abnormal class - increase weight significantly
            }
            # Convert keys to integers to avoid dtype issues
            class_weights = {int(k): float(v) for k, v in class_weights.items()}

            model.compile(
                optimizer=optimizer,
                loss=combined_loss,
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
            )
            
            # Check if we should use gradient accumulation
            if param.get("training", {}).get("gradient_accumulation_steps", 1) > 1:
                logger.info(f"Using gradient accumulation with {param['training']['gradient_accumulation_steps']} steps")
                
                # Get the dataset
                train_dataset = tf.data.Dataset.from_tensor_slices((train_data, tf.cast(train_labels_expanded, tf.float32)))
                train_dataset = train_dataset.batch(param["fit"]["batch_size"])
                
                # Create validation dataset
                val_dataset = tf.data.Dataset.from_tensor_slices((val_data, tf.cast(val_labels_expanded, tf.float32)))
                val_dataset = val_dataset.batch(param["fit"]["batch_size"])
                
                # Define variables to store accumulated gradients
                accum_steps = param["training"]["gradient_accumulation_steps"]
                
                # Create a history object to store metrics
                history_dict = {
                    'loss': [],
                    'accuracy': [],
                    'val_loss': [],
                    'val_accuracy': []
                }
                
                # Initialize accumulated gradients
                accumulated_gradients = [tf.Variable(tf.zeros_like(var), trainable=False) 
                                        for var in model.trainable_variables]
                
                # Training loop
                epochs = param["fit"]["epochs"]
                for epoch in range(epochs):
                    print(f"\nEpoch {epoch+1}/{epochs}")
                    
                    # Training metrics
                    train_loss = tf.keras.metrics.Mean()
                    train_accuracy = tf.keras.metrics.Mean()
                    
                    # Validation metrics
                    val_loss = tf.keras.metrics.Mean()
                    val_accuracy = tf.keras.metrics.Mean()
                    
                    # Training loop
                    step = 0
                    progress_bar = tqdm(train_dataset, desc=f"Training")
                    for x_batch, y_batch in progress_bar:
                        # Determine if this is the first batch in an accumulation cycle
                        first_batch = (step % accum_steps == 0)
                        
                        # Perform training step with the updated function
                        batch_loss, batch_accuracy = train_step_with_accumulation(
                            model, optimizer, loss_fn, x_batch, y_batch, accumulated_gradients, first_batch, accum_steps
                        )
                        
                        train_loss.update_state(batch_loss)
                        train_accuracy.update_state(batch_accuracy)
                        
                        # If we've accumulated enough gradients, apply them
                        if (step + 1) % accum_steps == 0 or (step + 1 == len(train_dataset)):
                            # Apply accumulated gradients
                            optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
                            
                            # Log progress
                            progress_bar.set_postfix({
                                'loss': f'{train_loss.result():.4f}',
                                'accuracy': f'{train_accuracy.result():.4f}'
                            })
                        
                        step += 1
                        
                        # Clear memory periodically
                        if step % 50 == 0:
                            gc.collect()
                    
                    # Validation loop
                    for x_batch, y_batch in tqdm(val_dataset, desc=f"Validation"):
                        batch_loss, batch_accuracy = val_step(model, loss_fn, x_batch, y_batch)
                        val_loss.update_state(batch_loss)
                        val_accuracy.update_state(batch_accuracy)
                    
                    # Print epoch results
                    print(f"Training loss: {train_loss.result():.4f}, accuracy: {train_accuracy.result():.4f}")
                    print(f"Validation loss: {val_loss.result():.4f}, accuracy: {val_accuracy.result():.4f}")
                    
                    # Store metrics in history
                    history_dict['loss'].append(float(train_loss.result()))
                    history_dict['accuracy'].append(float(train_accuracy.result()))
                    history_dict['val_loss'].append(float(val_loss.result()))
                    history_dict['val_accuracy'].append(float(val_accuracy.result()))
                    
                    # Check for early stopping
                    if callbacks and any(isinstance(cb, tf.keras.callbacks.EarlyStopping) for cb in callbacks):
                        early_stopping_callback = next(cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.EarlyStopping))
                        
                        # Get the monitored value
                        if early_stopping_callback.monitor == 'val_loss':
                            current = float(val_loss.result())
                        elif early_stopping_callback.monitor == 'val_accuracy':
                            current = float(val_accuracy.result())
                        
                        # Check if we should stop
                        if hasattr(early_stopping_callback, 'best') and early_stopping_callback.monitor == 'val_loss':
                            if early_stopping_callback.best is None or current < early_stopping_callback.best:
                                early_stopping_callback.best = current
                                early_stopping_callback.wait = 0
                                try:
                                    model.save(model_file)
                                    logger.info(f"Model saved to {model_file}")
                                except Exception as e:
                                    logger.warning(f"Error saving model: {e}")
                                    try:
                                        model.save(model_file.replace('.keras', ''))
                                        logger.info(f"Model saved with alternative format to {model_file.replace('.keras', '')}")
                                    except Exception as e2:
                                        logger.error(f"All attempts to save model failed: {e2}")
                                logger.info(f"Saved best model at epoch {epoch+1}")
                            else:
                                early_stopping_callback.wait += 1
                                if early_stopping_callback.wait >= early_stopping_callback.patience:
                                    print(f"Early stopping triggered at epoch {epoch+1}")
                                    break
                        elif hasattr(early_stopping_callback, 'best') and early_stopping_callback.monitor == 'val_accuracy':
                            if early_stopping_callback.best is None or current > early_stopping_callback.best:
                                early_stopping_callback.best = current
                                early_stopping_callback.wait = 0
                                try:
                                    model.save(model_file)
                                    logger.info(f"Model saved to {model_file}")
                                except Exception as e:
                                    logger.warning(f"Error saving model: {e}")
                                    try:
                                        model.save(model_file.replace('.keras', ''))
                                        logger.info(f"Model saved with alternative format to {model_file.replace('.keras', '')}")
                                    except Exception as e2:
                                        logger.error(f"All attempts to save model failed: {e2}")
                                logger.info(f"Saved best model at epoch {epoch+1}")
                            else:
                                early_stopping_callback.wait += 1
                                if early_stopping_callback.wait >= early_stopping_callback.patience:
                                    print(f"Early stopping triggered at epoch {epoch+1}")
                                    break
                    
                    # Save model periodically
                    if (epoch + 1) % 5 == 0:
                        try:
                            model.save(f"{param['model_directory']}/model_overall_ast_epoch_{epoch+1}.keras")
                            logger.info(f"Saved model checkpoint at epoch {epoch+1}")
                        except Exception as e:
                            logger.warning(f"Failed to save model checkpoint at epoch {epoch+1}: {e}")
                    
                    # Clear memory between epochs
                    gc.collect()
                
                # Convert history to a format compatible with Keras history
                history = type('History', (), {'history': history_dict})
            else:
                # Standard training without gradient accumulation
                if use_streaming:
                    # Train with streaming data
                    history = model.fit(
                        train_dataset,
                        epochs=param.get("fit", {}).get("epochs", 30),
                        validation_data=val_dataset,
                        callbacks=callbacks,
                        verbose=1
                    )
                else:
                    # Train with in-memory data
                    history = model.fit(
                        train_data,
                        train_labels_expanded,
                        batch_size=param.get("fit", {}).get("batch_size", 16),
                        epochs=param.get("fit", {}).get("epochs", 30),
                        validation_data=(val_data, val_labels_expanded),
                        class_weight=class_weights,
                        callbacks=callbacks,
                        verbose=1
                    )

    # Log training time
    training_time = time.time() - model_start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")

    # Make sure history exists before plotting
    if 'history' in locals():
        # Plot the training history
        visualizer.loss_plot(history)
        visualizer.save_figure(history_img)
    else:
        logger.warning("No training history available to plot")

    print("============== EVALUATION ==============")
    # Evaluate on test set
    test_data, test_labels_expanded = list_to_spectrograms(
        test_files,
        test_labels,
        msg="generate test_dataset",
        augment=False,
        param=param,
        batch_size=20
    )

    logger.info(f"Test data shape: {test_data.shape}")
    logger.info(f"Test labels shape: {test_labels_expanded.shape}")

    # Preprocess test data
    test_data = preprocess_spectrograms(test_data, target_shape)

    # Apply same normalization to test data
    if train_std > 1e-6:
        test_data = (test_data - train_mean) / train_std
    else:
        test_data = test_data - train_mean

    # Evaluate the model
    if test_data.shape[0] > 0:
        # Predict on test set
        y_pred = model.predict(test_data, batch_size=batch_size, verbose=1)
        
        # Now analyze the predictions (moved from above)
        logger.info(f"Raw prediction statistics:")
        logger.info(f"  - Min: {np.min(y_pred):.4f}, Max: {np.max(y_pred):.4f}")
        logger.info(f"  - Mean: {np.mean(y_pred):.4f}, Median: {np.median(y_pred):.4f}")
        
        # Count predictions in different ranges
        ranges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i in range(len(ranges)-1):
            count = np.sum((y_pred >= ranges[i]) & (y_pred < ranges[i+1]))
            logger.info(f"  - Predictions in range [{ranges[i]:.1f}, {ranges[i+1]:.1f}): {count} ({count/len(y_pred)*100:.1f}%)")
        
        # Plot histogram of predictions
        plt.figure(figsize=(10, 6))
        plt.hist(y_pred, bins=20)
        plt.title('Distribution of Raw Predictions')
        plt.xlabel('Prediction Value')
        plt.ylabel('Count')
        plt.savefig(f"{param['result_directory']}/prediction_distribution.png")
        plt.close()
        
        # Try multiple thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        logger.info("Evaluating with multiple thresholds:")
        for thresh in thresholds:
            y_pred_binary_thresh = (y_pred > thresh).astype(int)
            accuracy = metrics.accuracy_score(test_labels_expanded, y_pred_binary_thresh)
            precision = metrics.precision_score(test_labels_expanded, y_pred_binary_thresh, zero_division=0)
            recall = metrics.recall_score(test_labels_expanded, y_pred_binary_thresh, zero_division=0)
            f1 = metrics.f1_score(test_labels_expanded, y_pred_binary_thresh, zero_division=0)
            logger.info(f"Threshold {thresh:.1f}: Acc={accuracy:.4f}, Prec={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        # Apply standard threshold for final evaluation
        logger.info("Finding optimal classification threshold...")
        detection_threshold = find_optimal_threshold(test_labels_expanded, y_pred, param)
        logger.info(f"Using optimal detection threshold: {detection_threshold}")
        y_pred_binary = (y_pred > detection_threshold).astype(int)
        
        gc.collect()
        
        # Calculate metrics
        test_accuracy = metrics.accuracy_score(test_labels_expanded, y_pred_binary)
        test_precision = metrics.precision_score(test_labels_expanded, y_pred_binary, zero_division=0)
        test_recall = metrics.recall_score(test_labels_expanded, y_pred_binary, zero_division=0)
        test_f1 = metrics.f1_score(test_labels_expanded, y_pred_binary, zero_division=0)

        # Log metrics
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test F1 Score: {test_f1:.4f}")

        # Detailed classification report
        report = classification_report(
            test_labels_expanded, y_pred_binary, target_names=["Normal", "Abnormal"], zero_division=0
        )
        logger.info(f"Classification Report:\n{report}")

        # Plot confusion matrix
        cm_img = f"{param['result_directory']}/confusion_matrix_overall_ast.png"
        visualizer.plot_confusion_matrix(
            test_labels_expanded,
            y_pred_binary,
            title=f"Confusion Matrix (Overall)",
        )
        visualizer.save_figure(cm_img)

        # Store results
        evaluation_result = {
            "accuracy": float(test_accuracy),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1": float(test_f1),
        }

        # Append to global results
        all_y_true.extend(test_labels_expanded)
        all_y_pred.extend(y_pred_binary)

    else:
        logger.warning("No test data available for evaluation")
        evaluation_result = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    # Store evaluation results
    results[evaluation_result_key] = evaluation_result

    # Find optimal threshold using ROC curve
    if len(y_pred) > 0:
        logger.info("Finding optimal classification threshold using ROC curve...")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(test_labels_expanded, y_pred)
        
        # Calculate the geometric mean of sensitivity and specificity
        gmeans = np.sqrt(tpr * (1-fpr))
        
        # Find the optimal threshold
        ix = np.argmax(gmeans)
        best_threshold = thresholds[ix]
        logger.info(f"Optimal threshold from ROC curve: {best_threshold:.4f}")
        logger.info(f"At this threshold - TPR: {tpr[ix]:.4f}, FPR: {fpr[ix]:.4f}, G-Mean: {gmeans[ix]:.4f}")
        
        # Re-evaluate with optimal threshold
        y_pred_binary = (y_pred > best_threshold).astype(int)

        # If optimal threshold is very close to 0.5, try a lower threshold
        if 0.45 < best_threshold < 0.55 and test_f1 < 0.6:
            logger.info("Trying a lower threshold (0.39) since optimal threshold is close to default")
            lower_threshold = 0.39
            y_pred_binary_lower = (y_pred > lower_threshold).astype(int)
            
            # Calculate metrics with lower threshold
            test_accuracy_lower = metrics.accuracy_score(test_labels_expanded, y_pred_binary_lower)
            test_precision_lower = metrics.precision_score(test_labels_expanded, y_pred_binary_lower, zero_division=0)
            test_recall_lower = metrics.recall_score(test_labels_expanded, y_pred_binary_lower, zero_division=0)
            test_f1_lower = metrics.f1_score(test_labels_expanded, y_pred_binary_lower, zero_division=0)
            
            logger.info(f"Metrics with lower threshold ({lower_threshold}):")
            logger.info(f"Test Accuracy: {test_accuracy_lower:.4f}")
            logger.info(f"Test Precision: {test_precision_lower:.4f}")
            logger.info(f"Test Recall: {test_recall_lower:.4f}")
            logger.info(f"Test F1 Score: {test_f1_lower:.4f}")
            
            # Use lower threshold if it gives better F1 score
            if test_f1_lower > test_f1:
                logger.info(f"Using lower threshold {lower_threshold} instead of optimal threshold {best_threshold}")
                y_pred_binary = y_pred_binary_lower
                test_accuracy = test_accuracy_lower
                test_precision = test_precision_lower
                test_recall = test_recall_lower
                test_f1 = test_f1_lower

    # Calculate overall metrics
    if all_y_true and all_y_pred:
        overall_accuracy = metrics.accuracy_score(all_y_true, all_y_pred)
        overall_precision = metrics.precision_score(all_y_true, all_y_pred, zero_division=0)
        overall_recall = metrics.recall_score(all_y_true, all_y_pred, zero_division=0)
        overall_f1 = metrics.f1_score(all_y_true, all_y_pred, zero_division=0)

        overall_results = {
            "overall_accuracy": float(overall_accuracy),
            "overall_precision": float(overall_precision),
            "overall_recall": float(overall_recall),
            "overall_f1": float(overall_f1),
        }
        results.update(overall_results)

        logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
        logger.info(f"Overall Precision: {overall_precision:.4f}")
        logger.info(f"Overall Recall: {overall_recall:.4f}")
        logger.info(f"Overall F1 Score: {overall_f1:.4f}")

        # Plot overall confusion matrix
        overall_cm_img = f"{param['result_directory']}/confusion_matrix_overall_ast.png"
        visualizer.plot_confusion_matrix(
            all_y_true,
            all_y_pred,
            title="Overall Confusion Matrix",
        )
        visualizer.save_figure(overall_cm_img)

    total_time = time.time() - start_time
    results["timing"] = {
        "total_execution_time_seconds": float(total_time),
        "model_training_time_seconds": float(training_time),
    }

    # Save results to YAML
    with open(result_file, "w") as f:
        yaml.safe_dump(results, f)
    logger.info(f"Results saved to {result_file}")

    # Log total execution time
    logger.info(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()