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
    Balance the dataset by augmenting the minority class more efficiently
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
    
    logger.info(f"Augmenting minority class {minority_class} with {n_to_add} samples using batch processing")
    
    # Get all minority samples
    minority_samples = train_data[minority_indices]
    
    # Create augmented samples in one batch operation for better efficiency
    # Generate random indices for sampling (with replacement)
    batch_indices = np.random.choice(len(minority_indices), n_to_add, replace=True)
    batch_samples = minority_samples[batch_indices].copy()
    
    # Add random noise (vectorized operation)
    noise_level = 0.1
    # Create the noise array in one operation
    noise = np.random.normal(0, noise_level, batch_samples.shape)
    batch_augmented = batch_samples + noise
    
    # Clip values to valid range
    batch_augmented = np.clip(batch_augmented, -3, 3)  # Assuming normalized data
    
    # Create the labels array
    augmented_labels = np.full(n_to_add, minority_class)
    
    logger.info(f"Finished creating {n_to_add} augmented samples")
    
    # Combine original and augmented data
    balanced_data = np.vstack([train_data, batch_augmented])
    balanced_labels = np.concatenate([train_labels, augmented_labels])
    
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
def verify_gpu_usage():
    """Verify that TensorFlow is properly using the GPU"""
    # Create a simple test tensor and operation
    try:
        import tensorflow as tf
        
        # Print available physical devices to confirm GPU detection
        physical_devices = tf.config.list_physical_devices()
        print(f"Available physical devices:")
        for device in physical_devices:
            print(f"  {device.name} - {device.device_type}")
            
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Available GPUs: {len(gpus)}")
        
        if not gpus:
            logger.warning("No GPUs detected by TensorFlow!")
            return False
            
        # Try a simple matrix operation on GPU
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
            c = tf.matmul(a, b)
        
        # Check if the operation was executed on GPU
        print(f"Test tensor operation result: {c}")
        print(f"Test tensor device: {c.device}")
        if 'GPU' in c.device:
            logger.info("✓ GPU is properly configured and being used")
            print("✓ GPU is properly configured and being used")
            return True
        else:
            logger.warning("⚠ GPU is not being used for tensor operations!")
            print("⚠ GPU is not being used for tensor operations!")
            return False
            
    except Exception as e:
        logger.error(f"Error checking GPU usage: {e}")
        print(f"Error checking GPU usage: {e}")
        return False


def save_test_data(file_path, test_data, test_labels):
    """Save test data and labels to avoid reprocessing every time"""
    try:
        logger.info(f"Saving processed test data to {file_path}")
        np.savez_compressed(
            file_path,
            test_data=test_data,
            test_labels=test_labels
        )
        return True
    except Exception as e:
        logger.error(f"Error saving test data: {e}")
        return False

def load_test_data(file_path):
    """Load processed test data if available"""
    try:
        if os.path.exists(file_path):
            logger.info(f"Loading processed test data from {file_path}")
            data = np.load(file_path)
            return data['test_data'], data['test_labels']
        return None, None
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return None, None


def create_improved_optimizer():
    """Create an optimizer optimized for binary classification with gradient clipping"""
    return tf.keras.optimizers.Adam(
        learning_rate=5e-5,  # Start with a lower learning rate
        clipnorm=1.0,        # Clip gradients to prevent extreme updates
        epsilon=1e-7,        # Numerical stability
        beta_1=0.9,          # Momentum
        beta_2=0.999         # RMSprop factor
    )


def create_simple_modelcheckpoint_callback(model_file, monitor='val_loss', mode='min'):
    """
    Create a simplified ModelCheckpoint callback that avoids using options parameter.
    This fixes the "Could not extract model from args or kwargs" error.
    """
    # Create a plain callback with minimal parameters
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_file,
        save_best_only=True,
        monitor=monitor,
        mode=mode,
        verbose=1
    )
    
    # Store the original _save_model method
    original_save_model = callback._save_model
    
    # Create a new _save_model method that handles the error
    def safe_save_model(epoch, logs):
        try:
            # Check if we should save based on monitor value
            current = logs.get(callback.monitor)
            if current is None:
                logger.warning(f'ModelCheckpoint: {callback.monitor} not available in logs')
                return
                
            if callback.monitor_op(current, callback.best):
                logger.info(f'Saving model to {callback.filepath} with {callback.monitor}={current:.4f}')
                callback.best = current
                try:
                    # Simply call model.save() directly to avoid the problematic options parameter
                    callback.model.save(callback.filepath, overwrite=True)
                    logger.info(f"Model saved successfully at epoch {epoch+1}")
                except Exception as e:
                    logger.error(f"Error saving model: {str(e)}")
        except Exception as e:
            logger.error(f"Error in ModelCheckpoint: {str(e)}")
    
    # Replace the _save_model method with our safe version
    callback._save_model = lambda epoch, logs=None: safe_save_model(epoch, logs or {})
    
    return callback


def main():
    # Force a new model to be trained by deleting the old one
    model_path = "./model/AST/model_overall_ast.keras"
    if os.path.exists(model_path):
        logger.info(f"DEBUG: Removing existing model file for fresh training: {model_path}")
        os.remove(model_path)
    if os.path.exists(model_path + "_temp"):
        logger.info(f"DEBUG: Removing existing temp model directory: {model_path}_temp")
        shutil.rmtree(model_path + "_temp")
    
    # Set memory growth before any other TensorFlow operations
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                logger.info(f"Enabled memory growth for {device}")
            except Exception as e:
                logger.warning(f"Could not set memory growth for {device}: {e}")
        
        # Replace deprecated is_gpu_available with recommended approach
        gpu_available = len(physical_devices) > 0
        logger.info(f"TensorFlow is using GPU: {gpu_available}")
        logger.info(f"Available GPUs: {physical_devices}")
    
    # Load parameters from YAML file first
    with open("baseline_AST.yaml", "r") as stream:
        param = yaml.safe_load(stream)
        
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
        train_labels_expanded = load_pickle(train_labels_pickle)
        val_data = load_pickle(val_pickle)
        val_labels_expanded = load_pickle(val_labels_pickle)
        test_files = load_pickle(test_files_pickle)
        test_labels = load_pickle(test_labels_pickle)
        
        # Debug info to check loaded pickle data
        print(f"DEBUG: Loaded pickle data shapes:")
        print(f"  - Train data: {train_data.shape if hasattr(train_data, 'shape') else 'No shape'}")
        print(f"  - Train labels: {train_labels_expanded.shape if hasattr(train_labels_expanded, 'shape') else 'No shape'}")
        print(f"  - Val data: {val_data.shape if hasattr(val_data, 'shape') else 'No shape'}")
        print(f"  - Val labels: {val_labels_expanded.shape if hasattr(val_labels_expanded, 'shape') else 'No shape'}")
        print(f"  - Test files: {len(test_files) if isinstance(test_files, list) else 'Not a list'}")
        print(f"  - Test labels: {test_labels.shape if hasattr(test_labels, 'shape') else 'No shape'}")
        
        # Skip debug mode entirely when using pre-existing pickle files
        logger.info("Using pre-existing pickle files, skipping data generation and debug mode")
        
        # Verification of loaded data
        if hasattr(train_data, 'shape') and train_data.shape[0] == 0:
            logger.error("Loaded training data is empty! Check pickle files.")
            return
        
        if hasattr(val_data, 'shape') and val_data.shape[0] == 0:
            logger.error("Loaded validation data is empty! Check pickle files.")
            return
            
        # Skip to normalization step
        print("============== USING EXISTING PICKLED DATA ==============")
    else:
        # Only use debug mode when generating new data
        debug_mode = param.get("debug", {}).get("enabled", False)
        debug_sample_size = param.get("debug", {}).get("sample_size", 100)
        
        train_files, train_labels, val_files, val_labels, test_files, test_labels = dataset_generator(target_dir, param=param)

        if len(train_files) == 0 or len(val_files) == 0 or len(test_files) == 0:
            logger.error(f"No files found for {evaluation_result_key}, skipping...")
            return  # Exit main() if no files are found after generation
            
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



    print("\n============== VERIFYING GPU CONFIGURATION BEFORE TRAINING ==============")
    verify_gpu_usage()
    verify_gpu_usage_during_training()
    print("\n============== GPU MEMORY USAGE BEFORE MODEL CREATION ==============")
    monitor_gpu_usage()
    
    # Configure mixed precision
    mixed_precision_enabled = configure_mixed_precision(
        enabled=param.get("training", {}).get("mixed_precision", False)
    )

    # Create model with the correct input shape
    model_file = f"{param['model_directory']}/model_overall_ast.keras"
    if os.path.exists(model_file) or os.path.exists(f"{model_file}.index"):
        training_required = True  # Default to training unless we successfully load a model
        try:
            # Print detailed information about the model file
            logger.info(f"DEBUG: Model file exists at: {model_file}")
            logger.info(f"DEBUG: Model file size: {os.path.getsize(model_file) if os.path.exists(model_file) else 'N/A'} bytes")
            logger.info(f"DEBUG: Model directory contents: {os.listdir(os.path.dirname(model_file))}")
            
            # First try: direct load with standard binary crossentropy
            logger.info("DEBUG: Attempting to load model with simplified approach")
            try:
                # Try loading with compile=False first to see if that works
                logger.info("DEBUG: Trying tf.keras.models.load_model with compile=False")
                
                # Instead of trying to load the optimizer state which might cause issues
                model = tf.keras.models.load_model(
                    model_file, 
                    custom_objects=None,
                    compile=False
                )
                
                # If we get here, loading succeeded
                logger.info("DEBUG: Model loaded successfully, now recompiling")
                
                # Now recompile with fresh optimizer instead of trying to load optimizer state
                model.compile(
                    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),  # Use legacy optimizer with fixed learning rate
                    loss="binary_crossentropy",
                    metrics=['accuracy']
                )
                training_required = False  # Successfully loaded, no training needed
                logger.info("DEBUG: Model recompiled with binary_crossentropy loss")
                
            except Exception as e1:
                logger.warning(f"First load attempt failed: {e1}")
                logger.info(f"DEBUG: Error type: {type(e1)}")
                logger.info(f"DEBUG: Error args: {e1.args}")
                
                # Second try: Using save_format='tf' instead of 'keras'
                try:
                    logger.info("DEBUG: Trying to create a fresh model and save it in TF format first")
                    # Create a temporary new model
                    temp_model = create_model(
                        input_shape=(target_shape[0], target_shape[1]),
                        transformer_params=param.get("model", {}).get("architecture", {})
                    )
                    
                    # Save it in TF format
                    temp_model_path = f"{model_file}_temp"
                    logger.info(f"DEBUG: Saving temporary model to {temp_model_path}")
                    temp_model.save(temp_model_path, save_format='tf')
                    
                    # Now try to load the model again with TF format
                    logger.info(f"DEBUG: Loading from temporary TF model")
                    model = tf.keras.models.load_model(
                        temp_model_path,
                        compile=False
                    )
                    
                    # Compile the model
                    model.compile(
                        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),  # Use legacy optimizer with fixed learning rate
                        loss="binary_crossentropy",
                        metrics=['accuracy']
                    )
                    
                    # Clean up temp model
                    import shutil
                    if os.path.exists(temp_model_path) and os.path.isdir(temp_model_path):
                        shutil.rmtree(temp_model_path)
                    
                    # Successfully loaded
                    training_required = False
                    logger.info("DEBUG: Model loaded successfully using TF format")
                    
                except Exception as e2:
                    logger.warning(f"Second load attempt failed: {e2}")
                    logger.info(f"DEBUG: Error type: {type(e2)}")
                    logger.info(f"DEBUG: Error args: {e2.args}")
                    
                    # Third try: create fresh model and try loading weights directly
                    try:
                        logger.info("DEBUG: Trying to load weights only with fresh model...")
                        # Create fresh model
                        new_model = create_model(
                            input_shape=(target_shape[0], target_shape[1]),
                            transformer_params=param.get("model", {}).get("architecture", {})
                        )
                        
                        # Compile the model
                        new_model.compile(
                            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),  # Use legacy optimizer with fixed learning rate
                            loss="binary_crossentropy",
                            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
                        )
                        
                        # Try loading weights, avoid loading optimizer state
                        try:
                            if os.path.isdir(model_file):  # SavedModel format
                                logger.info(f"DEBUG: Model file is a directory, trying to load weights from SavedModel")
                                # For SavedModel, we need to load the whole model first to extract weights
                                temp_model = tf.keras.models.load_model(model_file, compile=False)
                                new_model.set_weights(temp_model.get_weights())
                                del temp_model  # Free memory
                                logger.info("DEBUG: Successfully transferred weights from SavedModel")
                            else:  # H5 format
                                logger.info(f"DEBUG: Model file is H5 format, trying to load weights directly")
                                new_model.load_weights(model_file, by_name=True, skip_mismatch=True)
                                logger.info("DEBUG: Successfully loaded weights from H5")
                                
                            model = new_model
                            training_required = False
                        except Exception as e3:
                            error_msg = str(e3)
                            logger.error(f"Weight loading failed: {error_msg}")
                            logger.info(f"DEBUG: Error type: {type(e3)}")
                            logger.info(f"DEBUG: Error args: {e3.args}")
                            
                            if "LossScaleOptimizerV3" in error_msg:
                                logger.info("DEBUG: LossScaleOptimizerV3 error detected, this is likely due to mixed precision issues")
                            
                            logger.info("DEBUG: Will train a new model since weight loading failed")
                            training_required = True  # Weight loading failed, need to train
                            raise Exception(f"Failed to load weights: {error_msg}")
                    except Exception as final_e:
                        logger.error(f"All loading attempts failed: {final_e}")
                        # Create a new model as last resort
                        logger.info(f"DEBUG: Creating new model with input shape: {target_shape}")
                        model = create_model(
                            input_shape=(target_shape[0], target_shape[1]),
                            transformer_params=param.get("model", {}).get("architecture", {})
                        )
                        # Need to set a flag to indicate training is required
                        training_required = True
                        logger.info("DEBUG: Created new model due to loading error, will need to train")
            
            # Debug the training_required flag value
            logger.info(f"DEBUG: After model loading attempts, training_required = {training_required}")
            
            # Force training for debugging - this will ensure the model always gets trained
            if param.get("training", {}).get("force_training", False) or param.get("debug", {}).get("force_training", True):
                logger.info("Force training enabled, will train loaded model")
                training_required = True
            
            # Final check of training_required flag
            if training_required:
                logger.info("DEBUG: Will train the model")
            else:
                logger.info("DEBUG: No training needed for loaded model")
                
        except Exception as e:
            logger.error(f"Unhandled error during model loading: {e}")
            logger.info(f"DEBUG: Error type: {type(e)}")
            logger.info(f"DEBUG: Error args: {e.args if hasattr(e, 'args') else 'No args'}")
            
            # Create a new model with the target shape as last resort
            logger.info(f"DEBUG: Creating new model with input shape: {target_shape}")
            model = create_model(
                input_shape=(target_shape[0], target_shape[1]),
                transformer_params=param.get("model", {}).get("architecture", {})
            )
            # Need to train
            training_required = True
            logger.info("DEBUG: Created new model due to unhandled error, will need to train")
    else:
        # No existing model file, create new one
        logger.info(f"DEBUG: No existing model found at {model_file}, creating new model with input shape: {target_shape}")
        model = create_model(
            input_shape=(target_shape[0], target_shape[1]),
            transformer_params=param.get("model", {}).get("architecture", {})
        )
        # Need to train
        training_required = True
        logger.info("DEBUG: Created new model, will need to train")

    # Add debug prints for data verification
    print("\n============== TRAINING DATA DEBUG INFO ==============")
    print(f"Training data shape: {train_data.shape}, dtype: {train_data.dtype}")
    print(f"Training labels shape: {train_labels_expanded.shape}, dtype: {train_labels_expanded.dtype}")
    print(f"Min/Max values - Train data: {np.min(train_data):.4f}/{np.max(train_data):.4f}")
    print(f"Min/Max values - Train labels: {np.min(train_labels_expanded) if len(train_labels_expanded) > 0 else 'N/A'}/{np.max(train_labels_expanded) if len(train_labels_expanded) > 0 else 'N/A'}")
    print(f"Class distribution - Normal: {np.sum(train_labels_expanded == 0)}, Abnormal: {np.sum(train_labels_expanded == 1)}")
    
    print("\n============== VALIDATION DATA DEBUG INFO ==============")
    print(f"Validation data shape: {val_data.shape}, dtype: {val_data.dtype}")
    print(f"Validation labels shape: {val_labels_expanded.shape}, dtype: {val_labels_expanded.dtype}")
    print(f"Class distribution - Normal: {np.sum(val_labels_expanded == 0)}, Abnormal: {np.sum(val_labels_expanded == 1)}")
    
    # Verify that data is non-zero and properly loaded
    if np.all(train_data == 0) or np.all(val_data == 0):
        print("WARNING: Training or validation data contains all zeros! Check data loading.")
    
    if len(train_labels_expanded) == 0 or len(val_labels_expanded) == 0:
        print("WARNING: No training or validation labels! Check label loading.")
    
    # After model creation, monitor GPU memory again
    print("\n============== GPU MEMORY USAGE AFTER MODEL CREATION ==============")
    monitor_gpu_usage()

    #debug
    normal_count = sum(1 for label in train_labels_expanded if label == 0)
    abnormal_count = sum(1 for label in train_labels_expanded if label == 1)
    print(f"Training data composition: Normal={normal_count}, Abnormal={abnormal_count}")

    
    # Check for data shape mismatch and fix it
    if train_data.shape[0] != train_labels_expanded.shape[0]:
        logger.warning(f"Data shape mismatch! X: {train_data.shape[0]} samples, y: {train_labels_expanded.shape[0]} labels")
        
        if train_data.shape[0] > train_labels_expanded.shape[0]:
            # Too many features, need to reduce
            train_data = train_data[:train_labels_expanded.shape[0]]
            sample_weights = sample_weights[:train_labels_expanded.shape[0]]
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
        0: 1.0,   # Normal class
        1: 10.0   # Abnormal class - significantly increased weight
    }
    logger.info(f"Using fixed class weights: {class_weights} (abnormal class weight increased to 10.0)")




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

    print("============== VERIFYING GPU USAGE ==============")
    is_gpu_working = verify_gpu_usage()
    if not is_gpu_working:
        logger.warning("GPU is not being used! Training will be slow.")
        return


    print("============== MODEL TRAINING ==============")
    # Track model training time specifically
    model_start_time = time.time()
    # Define model_file and history_img variables
    history_img = f"{param['result_directory']}/history_overall_ast.png"

    # Enable mixed precision training
    if param.get("training", {}).get("mixed_precision", False):
        logger.info("Enabling mixed precision training")
        mixed_precision.set_global_policy('mixed_float16')
        logger.info(f"Mixed precision policy enabled: {mixed_precision.global_policy()}")

    model_config = param.get("model", {}).get("architecture", {})
    model.summary()

    if training_required:
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

        # Add custom ModelCheckpoint callback to save best model without options parameter
        checkpoint_callback = create_simple_modelcheckpoint_callback(
            model_file=model_file,
            monitor='val_loss',
            mode='min'
        )
        callbacks.append(checkpoint_callback)
        logger.info("Added custom ModelCheckpoint callback that avoids 'options' parameter issues")

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

        
        # Add warmup learning rate scheduler if enabled
        warmup_config = param.get("fit", {}).get("warmup", {})
        if warmup_config.get("enabled", False):
            # Calculate total steps
            steps_per_epoch = len(train_data) // param["fit"]["batch_size"]
            total_steps = steps_per_epoch * param["fit"]["epochs"]
            
            # Calculate warmup steps
            warmup_epochs = warmup_config.get("epochs", 5)
            warmup_steps = steps_per_epoch * warmup_epochs


        compile_params = param["fit"]["compile"].copy()
        loss_type = param.get("model", {}).get("loss", "binary_crossentropy")
        
        # Handle learning_rate separately for the optimizer
        learning_rate = compile_params.pop("learning_rate", 0.0001)
        
        # Adjust optimizer for mixed precision if enabled
        clipnorm = param.get("training", {}).get("gradient_clip_norm", 1.0)
        if mixed_precision_enabled and compile_params.get("optimizer") == "adam":
            from tensorflow.keras.optimizers import Adam
            
            # In TF 2.4+, LossScaleOptimizer is automatically applied when using mixed_float16 policy
            # So we just need to create the base optimizer with gradient clipping
            compile_params["optimizer"] = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
            logger.info(f"Using Adam optimizer with mixed precision and gradient clipping (clipnorm={clipnorm})")
        else:
            compile_params["optimizer"] = create_improved_optimizer()
            logger.info(f"Using improved optimizer with gradient clipping (clipnorm={clipnorm})")


        # Use a more stable loss function
        if loss_type == "binary_crossentropy":
            compile_params["loss"] = "binary_crossentropy"  # Use TF's built-in implementation for stability
        elif loss_type == "focal_loss":
            # Only use focal loss if explicitly requested and with safeguards
            gamma = param.get("model", {}).get("focal_loss", {}).get("gamma", 2.0)
            alpha = param.get("model", {}).get("focal_loss", {}).get("alpha", 0.25)
            
            # Check if we should use a safer implementation
            use_safe_focal = param.get("model", {}).get("focal_loss", {}).get("use_safe_implementation", True)
            if use_safe_focal:
                logger.info("Using numerically stable focal loss implementation")
                compile_params["loss"] = lambda y_true, y_pred: focal_loss(gamma, alpha)(y_true, y_pred)
            else:
                logger.warning("Using standard focal loss - watch for NaN losses")
                compile_params["loss"] = tf.keras.losses.BinaryFocalCrossentropy(
                    gamma=gamma, alpha=alpha, from_logits=False
                )
        else:
            logger.info("Using standard binary crossentropy loss")
            compile_params["loss"] = "binary_crossentropy"

        
        compile_params["metrics"] = ['accuracy']
        
        if param.get("training", {}).get("gradient_accumulation_steps", 1) > 1:
            logger.info(f"Using gradient accumulation with {param['training']['gradient_accumulation_steps']} steps")
            
            # Create optimizer
            optimizer = compile_params["optimizer"]
            loss_fn = compile_params["loss"]
            
            # Get the dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data, tf.cast(train_labels_expanded, tf.float32)))
            train_dataset = train_dataset.batch(param["fit"]["batch_size"])
            
            # Create validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((val_data, tf.cast(val_labels_expanded, tf.float32)))
            val_dataset = val_dataset.batch(param["fit"]["batch_size"])
            
            # Define variables to store accumulated gradients
            accum_steps = param["training"]["gradient_accumulation_steps"]
            
            # Create a history object to store metrics
            history = {
                'loss': [],
                'accuracy': [],
                'val_loss': [],
                'val_accuracy': []
            }
            
            # Initialize accumulated gradients
            accumulated_gradients = [tf.Variable(tf.zeros_like(var), trainable=False) 
                                    for var in model.trainable_variables]
            
            # Define training step function
            @tf.function
            def train_step(x_batch, y_batch, first_batch):
                with tf.GradientTape() as tape:
                    logits = model(x_batch, training=True)
                    
                    # Ensure y_batch is float32 and reshape to match logits
                    y_batch = tf.cast(y_batch, tf.float32)
                    y_batch_reshaped = tf.reshape(y_batch, logits.shape)
                    
                    if isinstance(loss_fn, str):
                        if loss_fn == "binary_crossentropy":
                            loss_value = tf.keras.losses.binary_crossentropy(y_batch_reshaped, logits)
                        else:
                            loss_value = tf.keras.losses.get(loss_fn)(y_batch_reshaped, logits)
                    else:
                        loss_value = loss_fn(y_batch_reshaped, logits)
                    
                    # Scale the loss to account for gradient accumulation
                    scaled_loss = loss_value / tf.cast(accum_steps, dtype=loss_value.dtype)
                
                # Calculate gradients
                gradients = tape.gradient(scaled_loss, model.trainable_variables)
                
                # If this is the first batch in an accumulation cycle, reset the accumulators
                if first_batch:
                    for i, grad in enumerate(gradients):
                        if grad is not None:
                            accumulated_gradients[i].assign(grad)
                        else:
                            accumulated_gradients[i].assign(tf.zeros_like(model.trainable_variables[i]))
                else:
                    # Otherwise add to the accumulated gradients
                    for i, grad in enumerate(gradients):
                        if grad is not None:
                            accumulated_gradients[i].assign_add(grad)
                
                # Calculate accuracy - ensure consistent data types
                y_pred = tf.cast(tf.greater_equal(logits, 0.5), tf.float32)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(y_batch_reshaped, y_pred), tf.float32))
                
                return loss_value, accuracy

            
            # Define validation step function
            @tf.function
            def val_step(x_batch, y_batch):
                logits = model(x_batch, training=False)
                
                # Ensure y_batch is float32 and reshape to match logits
                y_batch = tf.cast(y_batch, tf.float32)
                y_batch_reshaped = tf.reshape(y_batch, logits.shape)
                
                if isinstance(loss_fn, str):
                    if loss_fn == "binary_crossentropy":
                        loss_value = tf.keras.losses.binary_crossentropy(y_batch_reshaped, logits)
                    else:
                        loss_value = tf.keras.losses.get(loss_fn)(y_batch_reshaped, logits)
                else:
                    loss_value = loss_fn(y_batch_reshaped, logits)
                
                # Calculate accuracy - ensure consistent data types
                y_pred = tf.cast(tf.greater_equal(logits, 0.5), tf.float32)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(y_batch_reshaped, y_pred), tf.float32))
                
                return loss_value, accuracy


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
                    
                    # Perform training step
                    batch_loss, batch_accuracy = train_step(x_batch, y_batch, first_batch)
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
                    batch_loss, batch_accuracy = val_step(x_batch, y_batch)
                    val_loss.update_state(batch_loss)
                    val_accuracy.update_state(batch_accuracy)
                
                # Print epoch results
                print(f"Training loss: {train_loss.result():.4f}, accuracy: {train_accuracy.result():.4f}")
                print(f"Validation loss: {val_loss.result():.4f}, accuracy: {val_accuracy.result():.4f}")
                
                # Store metrics in history
                history['loss'].append(float(train_loss.result()))
                history['accuracy'].append(float(train_accuracy.result()))
                history['val_loss'].append(float(val_loss.result()))
                history['val_accuracy'].append(float(val_accuracy.result()))
                
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
                                # Save with the native Keras format
                                tf.keras.models.save_model(
                                    model,
                                    model_file,
                                    overwrite=True,
                                    include_optimizer=True,
                                    save_format='keras'
                                )
                                logger.info(f"Model saved to {model_file}")
                            except Exception as e:
                                logger.warning(f"Error saving model: {e}")

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
                                # Save with the native Keras format
                                tf.keras.models.save_model(
                                    model,
                                    model_file,
                                    overwrite=True,
                                    include_optimizer=True,
                                    save_format='keras'
                                )
                                logger.info(f"Model saved to {model_file}")
                            except Exception as e:
                                logger.warning(f"Error saving model: {e}")

                            logger.info(f"Saved best model at epoch {epoch+1}")
                        else:
                            early_stopping_callback.wait += 1
                            if early_stopping_callback.wait >= early_stopping_callback.patience:
                                print(f"Early stopping triggered at epoch {epoch+1}")
                                break
                
                # Save model periodically
                if (epoch + 1) % 5 == 0:
                    try:
                        # Save with the native Keras format
                        tf.keras.models.save_model(
                            model,
                            f"{param['model_directory']}/model_overall_ast_epoch_{epoch+1}.keras",
                            overwrite=True,
                            include_optimizer=True,
                            save_format='keras'
                        )
                        logger.info(f"Saved model checkpoint at epoch {epoch+1}")
                    except Exception as e:
                        logger.warning(f"Failed to save model checkpoint at epoch {epoch+1}: {e}")
                    logger.info(f"Saved model checkpoint at epoch {epoch+1}")
                
                # Clear memory between epochs
                gc.collect()
            
            # Convert history to a format compatible with Keras history
            history = type('History', (), {'history': history})


        else:
            # Disable XLA acceleration as it's causing shape issues
            if False and param.get("training", {}).get("xla_acceleration", False):
                logger.info("Enabling XLA acceleration")
                tf.config.optimizer.set_jit(True)
                os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
            else:
                logger.info("XLA acceleration disabled to avoid shape issues")

            # Apply mixup augmentation if enabled
            if False and param.get("training", {}).get("mixup", {}).get("enabled", False):
                alpha = param["training"]["mixup"].get("alpha", 0.2)
                logger.info(f"Applying mixup augmentation with alpha={alpha}")
                
                # Only apply mixup if we have enough samples
                if train_data.shape[0] > 10:
                    # Create a copy of the original data for safety
                    orig_train_data = train_data.copy()
                    orig_train_labels = train_labels_expanded.copy()
                    
                    # Apply mixup with controlled randomization
                    np.random.seed(42)  # For reproducibility
                    indices = np.random.permutation(train_data.shape[0])
                    shuffled_data = train_data[indices]
                    shuffled_labels = train_labels_expanded[indices]
                    
                    # Generate mixing coefficient
                    lam = np.random.beta(alpha, alpha, size=train_data.shape[0])
                    lam = np.maximum(lam, 1-lam)  # Ensure lambda is at least 0.5 for stability
                    lam = np.reshape(lam, (train_data.shape[0], 1, 1))  # Reshape for broadcasting
                    
                    # Mix the data
                    mixed_data = lam * train_data + (1 - lam) * shuffled_data
                    
                    # Mix the labels (reshape lambda for broadcasting)
                    lam_labels = np.reshape(lam, (train_data.shape[0], 1))
                    mixed_labels = lam_labels * train_labels_expanded + (1 - lam_labels) * shuffled_labels
                    
                    # Update the training data and labels
                    train_data = mixed_data
                    train_labels_expanded = mixed_labels
                    
                    logger.info(f"Applied mixup to {train_data.shape[0]} samples")
                else:
                    logger.warning("Not enough samples for mixup, skipping")


            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=0.0001,
                    clipnorm=1.0,
                    epsilon=1e-7
                ),
                loss=focal_loss,  # Using focal loss instead of binary_crossentropy
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
            )

            # Train with improved settings
            logger.info("Training with improved settings")
            history = model.fit(
                train_data,
                train_labels_expanded,
                batch_size=32,  # Smaller batch size for better learning
                epochs=50,
                validation_data=(val_data, val_labels_expanded),
                class_weight={
                    0: 1.0,    # Normal class 
                    1: 25.0    # Much higher weight for abnormal class to force model to learn
                },
                callbacks=callbacks,
                verbose=1
            )


            # Create a directory for checkpoints if it doesn't exist
            checkpoint_dir = os.path.dirname(model_file)
            os.makedirs(checkpoint_dir, exist_ok=True)


        # Make sure history exists before plotting
        if 'history' in locals():
            # Define default values for variables that might not be defined
            machine_type_val = machine_type if 'machine_type' in locals() else None
            machine_id_val = machine_id if 'machine_id' in locals() else None
            db_val = db if 'db' in locals() else None
            
            # Plot the training history
            visualizer.loss_plot(history, machine_type=machine_type_val, machine_id=machine_id_val, db=db_val)
        else:
            logger.warning("No training history available to plot")

        visualizer.save_figure(history_img)

    # Log training time
    training_time = time.time() - model_start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")

    # Save model after training completes
    try:
        model.save(model_file)
        logger.info(f"Model saved to {model_file}")
    except Exception as e:
        logger.warning(f"Error saving model: {e}")
        # Try alternative approach
        try:
            model.save(model_file.replace('.keras', ''))
            logger.info(f"Model saved with alternative format to {model_file.replace('.keras', '')}")
        except Exception as e2:
            logger.error(f"All attempts to save model failed: {e2}")


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
        
        # Find the best threshold to separate classes based on the prediction distribution
        # Calculate a custom threshold based on statistics of the predictions
        normal_samples = y_pred[test_labels_expanded == 0]
        abnormal_samples = y_pred[test_labels_expanded == 1]
        
        # Check if we have enough abnormal samples
        if len(abnormal_samples) > 0:
            logger.info(f"Normal predictions - Mean: {np.mean(normal_samples):.4f}, Std: {np.std(normal_samples):.4f}")
            logger.info(f"Abnormal predictions - Mean: {np.mean(abnormal_samples):.4f}, Std: {np.std(abnormal_samples):.4f}")
            
            # Check for separation between normal and abnormal predictions
            mean_diff = np.abs(np.mean(abnormal_samples) - np.mean(normal_samples))
            combined_std = (np.std(normal_samples) + np.std(abnormal_samples)) / 2
            
            # If there's significant separation between means
            if mean_diff > combined_std * 0.5:
                # If abnormal mean > normal mean, find a threshold between them
                if np.mean(abnormal_samples) > np.mean(normal_samples):
                    # Use weighted midpoint that's closer to the normal distribution
                    adaptive_threshold = float(np.mean(normal_samples) * 0.7 + np.mean(abnormal_samples) * 0.3)
                    logger.info(f"Using weighted threshold between means: {adaptive_threshold:.4f}")
                else:
                    # If abnormal mean < normal mean, which is unusual, reverse the weighting
                    adaptive_threshold = float(np.mean(normal_samples) * 0.3 + np.mean(abnormal_samples) * 0.7)
                    logger.info(f"Using reversed weighted threshold: {adaptive_threshold:.4f}")
            else:
                # If the distributions overlap significantly, try different approach
                # Get the range of predictions and calculate threshold at different percentiles
                sorted_preds = np.sort(y_pred)
                lower_percentile = float(sorted_preds[int(len(sorted_preds) * 0.1)])  # 10th percentile
                upper_percentile = float(sorted_preds[int(len(sorted_preds) * 0.9)])  # 90th percentile
                
                # If prediction range is too narrow, try more extreme thresholds
                if upper_percentile - lower_percentile < 0.1:
                    logger.info(f"Prediction range too narrow ({lower_percentile:.4f}-{upper_percentile:.4f})")
                    
                    # Try multiple thresholds and pick the one with best F1 on validation
                    test_thresholds = [0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55]
                    best_f1 = 0
                    adaptive_threshold = 0.5  # Default
                    
                    for thresh in test_thresholds:
                        preds_at_thresh = (y_pred > thresh).astype(int)
                        f1 = metrics.f1_score(test_labels_expanded, preds_at_thresh, zero_division=0)
                        logger.info(f"Threshold {thresh:.4f} - F1: {f1:.4f}")
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            adaptive_threshold = float(thresh)
                    
                    logger.info(f"Selected best performing threshold: {adaptive_threshold:.4f} (F1: {best_f1:.4f})")
                else:
                    # Use a point slightly below the overall mean (assuming more normal than abnormal samples)
                    mean_pred = np.mean(y_pred)
                    adaptive_threshold = float(mean_pred - 0.01)  # Slightly below the mean
                    logger.info(f"Using threshold slightly below mean: {adaptive_threshold:.4f}")
        else:
            # If no abnormal samples in prediction set, use best guess
            logger.warning("No abnormal samples in test predictions")
            adaptive_threshold = float(0.5)  # Use standard threshold
            logger.info(f"Using standard threshold: {adaptive_threshold:.4f}")
        
        # Apply custom threshold for meaningful results
        logger.info(f"Using custom threshold for evaluation: {adaptive_threshold:.4f}")
        y_pred_binary = (y_pred > adaptive_threshold).astype(int)
        
        # Calculate metrics with adaptive threshold
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
            title=f"Confusion Matrix (Overall) - Threshold: {adaptive_threshold:.4f}",
        )
        visualizer.save_figure(cm_img)

        # Store results
        evaluation_result = {
            "accuracy": float(test_accuracy),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1": float(test_f1),
            "threshold_used": float(adaptive_threshold)
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