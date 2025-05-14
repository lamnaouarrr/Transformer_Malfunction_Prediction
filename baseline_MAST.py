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



from pathlib import Path
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, Add, MultiHeadAttention, LayerNormalization, Reshape, Permute, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.losses import mse as mean_squared_error
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from skimage.metrics import structural_similarity as ssim
from transformers import TFViTModel, ViTConfig
from tensorflow.keras import layers  # Add explicit import for layers module
########################################################################

########################################################################
# version
########################################################################
__versions__ = "3.0.0"
########################################################################

def binary_cross_entropy_loss(y_true, y_pred):
    """
    Binary cross-entropy loss for autoencoder with improved memory efficiency
    """
    # Use TF's built-in binary_crossentropy for better memory efficiency
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss implementation for binary classification with imbalanced datasets.
    
    Args:
        gamma: Focusing parameter. Higher values increase focus on hard examples.
        alpha: Weighting factor for the positive class.
    
    Returns:
        A loss function that computes focal loss.
    """
    def loss_function(y_true, y_pred):
        # Clip predictions for numerical stability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Calculate focal weight
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma)
        
        # Apply alpha weighting
        alpha_weight = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        
        # Combine for final loss
        focal_loss = alpha_weight * focal_weight * cross_entropy
        
        return tf.reduce_mean(focal_loss)
    
    return loss_function



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
    os.makedirs("./logs/log_MAST", exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, filename="./logs/log_MAST/baseline_MAST.log")
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



def list_to_spectrograms(file_list, labels=None, msg="calc...", augment=False, param=None, batch_size=64):
    """
    Process a list of files into spectrograms with optional labels - memory optimized version
    """
    # Ensure param is not None
    param = param or {}
    
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
    
    # Fix the base_directory access to ensure it's correctly using the default value
    base_dir = param.get("base_directory", "./dataset")
    other_dir = Path(base_dir) / other_condition
    
    # Get files from the other condition directory
    other_files = list(other_dir.glob(f"*.{ext}"))
    
    # Process other directory files
    for file_path in other_files:
        filename = file_path.name
        parts = filename.split('_')
        
        if len(parts) >= 4:
            db = parts[1]
            machine_type = parts[2]
            id_part = parts[3]
            machine_id = id_part.split('-')[0] if '-' in id_part else id_part
            
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
def create_model(input_shape, transformer_params):
    # Extract transformer parameters
    num_heads = transformer_params.get("num_heads", 4)
    dim_feedforward = transformer_params.get("dim_feedforward", 512)
    num_encoder_layers = transformer_params.get("num_encoder_layers", 2)
    attention_dropout = transformer_params.get("attention_dropout", 0.1)
    dropout_rate = 0.2
    
    # Create input layer
    inputs = layers.Input(shape=input_shape)
    
    # Reshape to add channel dimension for Conv2D
    x = layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    # Patch embedding using Conv2D
    x = layers.Conv2D(filters=512, kernel_size=4, strides=4, padding='same', name='patch_embedding')(x)
    x = layers.BatchNormalization()(x)
    
    # Reshape for transformer
    batch_size = tf.shape(x)[0]
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]
    c = tf.shape(x)[3]
    x = layers.Reshape((h * w, c))(x)
    
    # Position encoding (optional but improves results)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Add position embeddings
    positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
    pos_encoding = positional_encoding(tf.shape(x)[1], tf.shape(x)[2])
    pos_encoding = pos_encoding[tf.newaxis, :, :]
    x = x + pos_encoding[:, :tf.shape(x)[1], :]
    
    # Transformer Encoder Layers
    for i in range(num_encoder_layers):
        # Multi-head attention block
        residual = x
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        mha = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=dim_feedforward // num_heads,
            dropout=attention_dropout,
            name=f'encoder_mha_{i}'
        )(x, x)
        x = layers.Add()([residual, mha])
        
        # Feed-forward block
        residual = x
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dense(dim_feedforward * 2)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dim_feedforward)(x)
        x = layers.Add()([residual, x])
    
    # Final normalization and global pooling
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head with stronger regularization
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='gelu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer with high-contrast activation
    # This is the key fix for the narrow prediction range issue
    raw_output = layers.Dense(1, activation=None)(x)
    
    # Add a lambda layer to increase prediction contrast 
    # This will help spread predictions further from 0.5 threshold
    outputs = layers.Lambda(
        lambda x: tf.sigmoid(5.0 * x),  # Multiply by 5 to increase contrast/separation
        name='contrast_sigmoid'
    )(raw_output)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_mast_model(input_shape, mast_params, transformer_params):
    """
    Creates a MAST (Masked Audio Spectrogram Transformer) model with two variants:
    1. A pretraining model that reconstructs masked inputs
    2. A fine-tuning model for anomaly detection
    
    Args:
        input_shape: Tuple of (height, width) for input spectrograms
        mast_params: MAST-specific parameters from config
        transformer_params: Transformer architecture parameters
        
    Returns:
        pretrain_model: Model for masked pretraining
        finetune_model: Model for anomaly detection fine-tuning
    """
    logger.info(f"Creating MAST model with input shape {input_shape}")
    
    # Extract parameters
    patch_size = mast_params.get("patch_size", 16)
    embed_dim = transformer_params.get("embed_dim", 768)
    num_heads = transformer_params.get("num_heads", 12)
    num_layers = transformer_params.get("num_layers", 12)
    mlp_dim = transformer_params.get("mlp_dim", 3072)
    dropout_rate = transformer_params.get("dropout_rate", 0.1)
    
    # Add explicit masking rate for pretraining
    mask_prob = mast_params.get("pretraining", {}).get("masking", {}).get("probability", 0.15)
    
    # Input layers
    inputs = layers.Input(shape=(*input_shape, 1))  # Add channel dimension
    
    # Patch embedding
    patch_height = min(patch_size, input_shape[0])
    patch_width = min(patch_size, input_shape[1])
    
    # Calculate number of patches
    num_patches_height = input_shape[0] // patch_height
    num_patches_width = input_shape[1] // patch_width
    total_patches = num_patches_height * num_patches_width
    
    logger.info(f"MAST: Using patch size {patch_height}x{patch_width} with {total_patches} total patches")
    
    # Create patches using Conv2D
    x = layers.Conv2D(
        filters=embed_dim,
        kernel_size=(patch_height, patch_width),
        strides=(patch_height, patch_width),
        padding="valid",
        name="patch_embedding"
    )(inputs)
    
    # Reshape to sequence
    batch_size = tf.shape(inputs)[0]
    x = layers.Reshape((total_patches, embed_dim))(x)
    
    # Add positional embedding
    positions = tf.range(start=0, limit=total_patches, delta=1)
    pos_embedding = layers.Embedding(
        input_dim=total_patches,
        output_dim=embed_dim,
        name="position_embedding"
    )(positions)
    
    # Add positional embedding to patches
    x = x + tf.expand_dims(pos_embedding, axis=0)  # (1, total_patches, embed_dim)
    
    # Add classification token ([CLS])
    cls_token = layers.Layer(name="cls_token")(
        tf.Variable(initial_value=tf.random.normal([1, 1, embed_dim]), trainable=True)
    )
    cls_tokens = tf.repeat(cls_token, repeats=batch_size, axis=0)
    x = tf.concat([cls_tokens, x], axis=1)  # (batch_size, total_patches + 1, embed_dim)
    
    # Apply dropout
    x = layers.Dropout(dropout_rate)(x)
    
    # Define block_size based on patch size
    block_size = patch_size // patch_height
    
    # Create a separate reconstruction head for pretraining
    reconstruction_head = layers.Dense(
        units=embed_dim,
        activation='gelu',
        name="reconstruction_dense_1"
    )
    reconstruction_head_2 = layers.Dense(
        units=patch_height * patch_width,
        name="reconstruction_output"
    )
    
    # Apply Transformer blocks
    for i in range(num_layers):
        # Normalization and Multi-Head Attention
        attn_output = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
            name=f"transformer_block_{i}_mha"
        )(attn_output, attn_output)
        x = layers.Add()([x, attn_output])
        
        # Normalization and MLP
        mlp_output = layers.LayerNormalization(epsilon=1e-6)(x)
        mlp_output = layers.Dense(mlp_dim, activation='gelu')(mlp_output)
        mlp_output = layers.Dropout(dropout_rate)(mlp_output)
        mlp_output = layers.Dense(embed_dim)(mlp_output)
        mlp_output = layers.Dropout(dropout_rate)(mlp_output)
        x = layers.Add()([x, mlp_output])
    
    # Final normalization
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Extract the [CLS] token output for classification (first token)
    cls_output = layers.Lambda(lambda x: x[:, 0], name="extract_cls")(x)
    
    # For pretraining model: Reconstruct the original input from all tokens (excluding CLS)
    reconstruction_tokens = layers.Lambda(lambda x: x[:, 1:], name="extract_tokens")(x)
    reconstructed = reconstruction_head(reconstruction_tokens)
    reconstructed = reconstruction_head_2(reconstructed)
    
    # Reshape back to image shape for reconstruction
    reconstructed = layers.Reshape((num_patches_height, num_patches_width, patch_height * patch_width))(reconstructed)
    
    # Adjust channel dimension to be divisible by block_size^2
    reconstructed = layers.Conv2D(
        filters=block_size**2,  # Ensure channel dimension matches block_size^2
        kernel_size=1,  # Use 1x1 convolution to adjust channels without affecting spatial dimensions
        padding="same",
        name="channel_adjustment"
    )(reconstructed)

    # Use depth-to-space (pixel shuffle) to go from patch embeddings back to full image
    reconstructed = layers.Lambda(
        lambda x: tf.nn.depth_to_space(
            tf.reshape(x, [
                tf.shape(x)[0],
                tf.shape(x)[1] * patch_height,
                tf.shape(x)[2] * patch_width,
                block_size**2
            ]),
            block_size=block_size
        ),
        name="reconstruction_reshape"
    )(reconstructed)
    
    # Create classifier head for anomaly detection
    classifier = layers.Dense(512, activation='gelu', name="classifier_dense_1")(cls_output)
    classifier = layers.Dropout(0.1)(classifier)
    classifier = layers.Dense(128, activation='gelu', name="classifier_dense_2")(classifier)
    classifier = layers.Dropout(0.1)(classifier)
    classifier = layers.Dense(1, activation='sigmoid', name="anomaly_output")(classifier)
    
    # Create two models: one for pretraining and one for fine-tuning
    pretrain_model = tf.keras.Model(inputs=inputs, outputs=reconstructed, name="MAST_pretrain")
    finetune_model = tf.keras.Model(inputs=inputs, outputs=classifier, name="MAST_finetune")
    
    return pretrain_model, finetune_model


class MaskingLayer(layers.Layer):
    """
    Layer that applies random masking to the input spectrogram for MAST pretraining.
    """
    def __init__(self, mask_probability=0.15, mask_length=8, mask_time=True, mask_freq=True, **kwargs):
        super(MaskingLayer, self).__init__(**kwargs)
        self.mask_probability = mask_probability
        self.mask_length = mask_length
        self.mask_time = mask_time
        self.mask_freq = mask_freq
        
    def call(self, inputs, training=None):
        if training:
            return self._apply_mask(inputs)
        return inputs
    
    def _apply_mask(self, x):
        shape = tf.shape(x)
        batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
        
        mask = tf.ones_like(x, dtype=tf.float32)
        
        # Time masking (horizontal)
        if self.mask_time:
            time_mask = tf.random.uniform(shape=[batch_size, self.mask_length, width, channels], minval=0, maxval=1)
            time_mask = tf.cast(time_mask < self.mask_probability, tf.float32)
            start_indices = tf.random.uniform(shape=[batch_size], minval=0, maxval=height-self.mask_length+1, dtype=tf.int32)
            
            for i in range(batch_size):
                start = start_indices[i]
                time_mask_part = time_mask[i:i+1]
                padding = [[0, 0], [start, height-self.mask_length-start], [0, 0], [0, 0]]
                padded_mask = tf.pad(time_mask_part, padding)
                mask = mask * (1.0 - padded_mask)
        
        # Frequency masking (vertical)
        if self.mask_freq:
            freq_mask = tf.random.uniform(shape=[batch_size, height, self.mask_length, channels], minval=0, maxval=1)
            freq_mask = tf.cast(freq_mask < self.mask_probability, tf.float32)
            start_indices = tf.random.uniform(shape=[batch_size], minval=0, maxval=width-self.mask_length+1, dtype=tf.int32)
            
            for i in range(batch_size):
                start = start_indices[i]
                freq_mask_part = freq_mask[i:i+1]
                padding = [[0, 0], [0, 0], [start, width-self.mask_length-start], [0, 0]]
                padded_mask = tf.pad(freq_mask_part, padding)
                mask = mask * (1.0 - padded_mask)
        
        # Apply mask (set masked positions to zero)
        masked_x = x * mask
        
        return masked_x
    
    def get_config(self):
        config = super(MaskingLayer, self).get_config()
        config.update({
            'mask_probability': self.mask_probability,
            'mask_length': self.mask_length,
            'mask_time': self.mask_time,
            'mask_freq': self.mask_freq,
        })
        return config


def apply_masking(x, mask_probability=0.15, mask_length=8, mask_time=True, mask_freq=True):
    """
    Applies masking to spectrograms for MAST pretraining.
    
    Args:
        x: Input spectrograms of shape (batch_size, height, width)
        mask_probability: Probability of masking
        mask_length: Length of mask segments
        mask_time: Whether to apply time masking
        mask_freq: Whether to apply frequency masking
    
    Returns:
        masked_x: Masked spectrograms
        mask: Binary mask (1 for kept, 0 for masked)
    """
    if len(x.shape) == 3:  # Add channel dimension if needed
        x = np.expand_dims(x, axis=-1)
    
    shape = x.shape
    batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    
    # Initialize mask with ones (all values kept)
    mask = np.ones_like(x, dtype=np.float32)
    
    # Apply time masking (horizontal)
    if mask_time:
        for i in range(batch_size):
            if np.random.random() < mask_probability:
                # Choose random start point and apply mask
                start = np.random.randint(0, height - mask_length + 1)
                mask[i, start:start+mask_length, :, :] = 0
    
    # Apply frequency masking (vertical)
    if mask_freq:
        for i in range(batch_size):
            if np.random.random() < mask_probability:
                # Choose random start point and apply mask
                start = np.random.randint(0, width - mask_length + 1)
                mask[i, :, start:start+mask_length, :] = 0
    
    # Apply mask
    masked_x = x * mask
    
    return masked_x, mask


def create_lr_schedule(initial_lr, warmup_epochs, decay_epochs):
    """
    Creates a learning rate schedule with warmup and decay.
    
    Args:
        initial_lr: Initial learning rate
        warmup_epochs: Number of warmup epochs
        decay_epochs: Number of epochs after which to decay to min_lr
    
    Returns:
        schedule_fn: Learning rate scheduler function
    """
    def schedule_fn(epoch):
        # Warmup phase
        if epoch < warmup_epochs:
            return initial_lr * ((epoch + 1) / warmup_epochs)
        
        # Decay phase
        decay_progress = (epoch - warmup_epochs) / (decay_epochs - warmup_epochs)
        decay_factor = 0.5 * (1 + tf.math.cos(tf.constant(math.pi) * tf.minimum(decay_progress, 1.0)))
        return initial_lr * decay_factor
    
    return schedule_fn


def preprocess_spectrograms(spectrograms, target_shape, param=None):
    """
    Resize all spectrograms to a consistent shape.
    """
    # Ensure param is not None
    param = param or {}
    
    # Handle case where input is a list of file paths instead of spectrograms
    spectrograms = list_to_spectrograms(spectrograms, None, "Processing files", False, param)
    
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
    Balance the dataset by augmenting the minority class with GPU acceleration
    V100 optimization: Uses batch operations for all augmentations
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
    
    logger.info(f"Augmenting minority class {minority_class} with {n_to_add} samples using GPU-optimized batch processing")
    
    # Get all minority samples
    minority_samples = train_data[minority_indices]
    
    # Process in batches to avoid memory issues
    # For V100 with 32GB, we can use larger batch sizes
    batch_size = 5000  # Optimized for V100 GPU
    n_batches = (n_to_add + batch_size - 1) // batch_size
    
    all_augmented = []
    
    # Try to use TensorFlow for GPU acceleration if available
    try:
        import tensorflow as tf
        use_tf = True
        logger.info("Using TensorFlow for GPU-accelerated augmentation")
    except ImportError:
        use_tf = False
        logger.info("TensorFlow not available, using NumPy for augmentation")
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_to_add)
        batch_size_actual = end_idx - start_idx
        
        # Generate random indices for sampling (with replacement)
        batch_indices = np.random.choice(len(minority_indices), batch_size_actual, replace=True)
        
        if use_tf:
            # TensorFlow operations for GPU acceleration
            # Convert to TensorFlow tensors
            batch_samples_tf = tf.convert_to_tensor(minority_samples[batch_indices], dtype=tf.float32)
            
            # Add random noise (vectorized operation)
            noise_level = 0.1
            noise_shape = tf.shape(batch_samples_tf)
            noise = tf.random.normal(noise_shape, mean=0.0, stddev=noise_level, dtype=tf.float32)
            batch_augmented = batch_samples_tf + noise
            
            # Apply random time/frequency shifts for more diversity
            if np.random.rand() > 0.5:
                # Random shift along time axis (dim 2)
                shift = np.random.randint(-3, 4)  # Shift by -3 to 3 steps
                if shift > 0:
                    batch_augmented = tf.pad(batch_augmented[:, :, :-shift], [[0, 0], [0, 0], [shift, 0]])
                elif shift < 0:
                    batch_augmented = tf.pad(batch_augmented[:, :, -shift:], [[0, 0], [0, 0], [0, -shift]])
            
            # Apply small random scaling
            scale_factor = tf.random.uniform([], 0.95, 1.05)
            batch_augmented = batch_augmented * scale_factor
            
            # Clip values to valid range
            batch_augmented = tf.clip_by_value(batch_augmented, -3.0, 3.0)  # Assuming normalized data
            
            # Convert back to numpy
            batch_augmented = batch_augmented.numpy()
        else:
            # NumPy operations as fallback
            batch_samples = minority_samples[batch_indices].copy()
            
            # Add random noise (vectorized operation)
            noise_level = 0.1
            noise = np.random.normal(0, noise_level, batch_samples.shape)
            batch_augmented = batch_samples + noise
            
            # Clip values to valid range
            batch_augmented = np.clip(batch_augmented, -3, 3)  # Assuming normalized data
        
        all_augmented.append(batch_augmented)
    
    # Combine all batches
    if all_augmented:
        batch_augmented = np.vstack(all_augmented)
        logger.info(f"Finished creating {batch_augmented.shape[0]} augmented samples")
        
        # Create the labels array
        augmented_labels = np.full(batch_augmented.shape[0], minority_class)
        
        # Combine original and augmented data
        balanced_data = np.vstack([train_data, batch_augmented])
        balanced_labels = np.concatenate([train_labels, augmented_labels])
        
        # Free memory
        del all_augmented, batch_augmented
        gc.collect()
        
        # Shuffle the data
        indices = np.arange(len(balanced_labels))
        np.random.shuffle(indices)
        balanced_data = balanced_data[indices]
        balanced_labels = balanced_labels[indices]
        
        logger.info(f"New dataset shape: {balanced_data.shape}")
        new_class_counts = dict(zip(*np.unique(balanced_labels, return_counts=True)))
        logger.info(f"New class distribution: {new_class_counts}")
        
        return balanced_data, balanced_labels
    else:
        logger.warning("No augmented samples created, returning original data")
        return train_data, train_labels


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
    
    return mixed_x, mixed_y


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
        import tensorflow as tf
        
        # Get GPU device information from TensorFlow
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.warning("No GPUs detected by TensorFlow")
            return None, None, None
            
        # Get GPU memory usage using nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                               stdout=subprocess.PIPE, text=True)
        memory_info = result.stdout.strip().split(',')
        
        # Parse the memory information
        used_memory = int(memory_info[0])
        total_memory = int(memory_info[1])
        gpu_utilization = int(memory_info[2]) if len(memory_info) > 2 else 0
        
        usage_percent = (used_memory / total_memory) * 100
        
        # Log detailed GPU information
        logger.info(f"GPU Memory: {used_memory}MB / {total_memory}MB ({usage_percent:.1f}%)")
        logger.info(f"GPU Utilization: {gpu_utilization}%")
        
        # Print to console as well for immediate visibility
        print(f"GPU Memory: {used_memory}MB / {total_memory}MB ({usage_percent:.1f}%)")
        print(f"GPU Utilization: {gpu_utilization}%")
        
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
    # Clip predictions for numerical stability
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate loss with higher weight for positive class
    loss = -(pos_weight * y_true * tf.math.log(y_pred) + 
            (1 - y_true) * tf.math.log(1 - y_pred))
    
    return tf.reduce_mean(loss)


# Create a learning rate scheduler with warmup
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


def verify_gpu_usage_during_training():
    """Monitor GPU usage during training to ensure the GPU is being used"""
    try:
        # Check GPU memory usage with nvidia-smi
        import subprocess
        import time
        
        # Print initial GPU status
        print("\n==== GPU STATUS BEFORE TRAINING OPERATIONS ====")
        subprocess.run(['nvidia-smi'], check=True)
        
        # Run a small test operation that should use the GPU
        print("\n==== PERFORMING TEST OPERATIONS ON GPU ====")
        import tensorflow as tf
        import numpy as np
        
        # Create two large tensors and perform matrix multiplication
        with tf.device('/GPU:0'):
            # Create large tensors to ensure GPU is utilized
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            
            # Time the operation
            start_time = time.time()
            # Perform multiple operations to ensure GPU utilization
            for _ in range(10):
                c = tf.matmul(a, b)
                c = tf.nn.relu(c)
            
            # Force evaluation
            result = c.numpy()
            end_time = time.time()
            
        print(f"Matrix operation completed in {end_time - start_time:.4f} seconds")
        print(f"Result shape: {result.shape}, device: {c.device}")
        
        # Check GPU memory usage after operation
        print("\n==== GPU STATUS AFTER TRAINING OPERATIONS ====")
        subprocess.run(['nvidia-smi'], check=True)
        
        # If the operation happened on GPU, it should be fast
        gpu_working = 'GPU' in c.device
        
        if gpu_working and (end_time - start_time) < 2.0:
            print("✓ GPU is properly working and performing calculations")
            return True
        elif gpu_working:
            print("⚠ GPU is being used but performing slowly")
            return True
        else:
            print("❌ Operations are not running on GPU")
            return False
        
    except Exception as e:
        print(f"Error during GPU verification: {e}")
        return False


########################################################################
# main
########################################################################
def main():
    # Load configurations
    with open('baseline_MAST.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Extract configurations
    model_params = config.get('model', {})
    mast_params = model_params.get('mast', {})
    transformer_params = model_params.get('architecture', {}).get('transformer', {})
    dataset_params = config.get('dataset', {})
    training_params = config.get('training', {})
    
    # Set up logging
    setup_logging()
    
    # Log configuration info
    logger.info(f"Starting MAST model training with config: {config}")
    
    # Check if we should load existing model or create a new one
    if training_params.get('load_model', False) and os.path.exists(model_params.get('model_path', '')):
        logger.info(f"Loading existing model from {model_params['model_path']}")
        model = tf.keras.models.load_model(model_params['model_path'])
    else:
        # Set random seeds for reproducibility
        tf.random.set_seed(training_params.get('random_seed', 42))
        np.random.seed(training_params.get('random_seed', 42))
        
        # Load and preprocess dataset
        logger.info("Loading dataset")
        # Fix: Use the normal directory to properly process both normal and abnormal data
        base_dir = config.get('base_directory', './dataset')
        normal_dir = os.path.join(base_dir, 'normal')
        
        train_files, train_labels, val_files, val_labels, test_files, test_labels = dataset_generator(
            normal_dir, config)
        
        # Get input shape from data
        target_shape = (config["feature"]["n_mels"], 96)
        logger.info(f"Target spectrogram shape: {target_shape}")
        
        # Preprocess to ensure consistent shapes
        logger.info("Preprocessing training data...")
        train_data = preprocess_spectrograms(train_files, target_shape)
        logger.info(f"Preprocessed train data shape: {train_data.shape}")

        logger.info("Preprocessing validation data...")
        val_data = preprocess_spectrograms(val_files, target_shape)
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
            pretrain_lr = mast_params.get('pretraining', {}).get('learning_rate', 1e-4)
            
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
                    filepath="model/MAST/pretrain_model.h5",
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
            pretrain_model.save_weights("model/MAST/pretrain_weights.h5")
            
            # Load the pretrained weights into the fine-tuning model
            # The shared Transformer layers will have the same names
            logger.info("Transferring pretrained weights to fine-tuning model")
            finetune_model.load_weights("model/MAST/pretrain_weights.h5", by_name=True, skip_mismatch=True)
        
        # Now proceed with fine-tuning for anomaly detection
        logger.info("Starting MAST fine-tuning phase for anomaly detection")
        
        # Configure fine-tuning optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=training_params.get('learning_rate', 0.0001))
        
        # Configure loss function based on configuration
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        
        # Compile model for fine-tuning
        finetune_model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        # Add channel dimension if needed
        if len(train_data.shape) == 3:
            train_data = np.expand_dims(train_data, axis=-1)
            val_data = np.expand_dims(val_data, axis=-1)
        
        # Set up callbacks for fine-tuning
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_params['model_path'],
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=training_params.get('early_stopping_patience', 10),
                restore_best_weights=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f"logs/log_mast/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                histogram_freq=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train the fine-tuning model
        history = finetune_model.fit(
            train_data, train_labels_expanded,
            validation_data=(val_data, val_labels),
            epochs=training_params.get('epochs', 100),
            batch_size=training_params.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        model = finetune_model
        model.save(model_params['model_path'])
        
        # Save training history
        with open('pickle/pickle_mast/training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)
    
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
    
    # Calculate metrics
    accuracy = metrics.accuracy_score(test_labels_expanded, y_pred_binary)
    precision = metrics.precision_score(test_labels_expanded, y_pred_binary)
    recall = metrics.recall_score(test_labels_expanded, y_pred_binary)
    f1 = metrics.f1_score(test_labels_expanded, y_pred_binary)
    auc = metrics.roc_auc_score(test_labels_expanded, y_pred)
    
    # Print metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC: {auc:.4f}")
    
    # Generate confusion matrix
    cm = metrics.confusion_matrix(test_labels_expanded, y_pred_binary)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Plot metrics
    plt.figure(figsize=(15, 5))
    
    # Plot ROC curve
    plt.subplot(1, 3, 1)
    fpr, tpr, _ = metrics.roc_curve(test_labels_expanded, y_pred)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Plot training history if available
    if os.path.exists('pickle/pickle_mast/training_history.pkl'):
        with open('pickle/pickle_mast/training_history.pkl', 'rb') as f:
            history = pickle.load(f)
        
        plt.subplot(1, 3, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('result/result_mast/performance_metrics.png')
    logger.info(f"Performance metrics saved to result/result_mast/performance_metrics.png")
    
    # Ensure necessary directories exist for saving artifacts
    result_dir = config.get('result_directory', './result/result_MAST')
    os.makedirs(result_dir, exist_ok=True)

    # Save the hyperparameters used in the model to a new YAML file
    hyperparams_file_path = os.path.join(result_dir, 'used_hyperparameters.yaml')
    with open(hyperparams_file_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    logger.info(f"Hyperparameters used in the model saved to {hyperparams_file_path}")

    # Remove unnecessary lines from the results dictionary
    results.pop('model_path', None)
    results.pop('result_file_path', None)

    # Save as YAML file
    yaml_file_path = os.path.join(result_dir, config.get('result_file', 'test_results.yaml'))
    with open(yaml_file_path, 'w') as f:
        yaml.safe_dump(results, f, default_flow_style=False)

    logger.info(f"Test results saved to {yaml_file_path}")
    
    return model


if __name__ == "__main__":
    main()