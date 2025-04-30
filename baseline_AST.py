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


from tensorflow.keras import mixed_precision
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, Add, MultiHeadAttention, LayerNormalization, Reshape, Permute, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.losses import mse as mean_squared_error
from tensorflow.keras.regularizers import l2
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
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

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal loss for addressing class imbalance with improved numerical stability
    """
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
        seq_len: sequence length (can be a tensor or int)
        d_model: depth of the model
        encoding_type: type of positional encoding
        
    Returns:
        Positional encoding matrix
    """
    # For dynamic tensors, use a simpler approach
    if tf.is_tensor(seq_len):
        # Create a fixed-size positional encoding that will work for any reasonable sequence length
        max_len = 1024  # Large enough for most applications
        
        # Create position vector
        position = tf.range(max_len, dtype=tf.float32)
        position = tf.expand_dims(position, axis=1)  # (max_len, 1)
        
        # Create dimension vector
        div_term = tf.range(0, d_model, 2, dtype=tf.float32)
        div_term = tf.math.pow(10000.0, div_term / tf.cast(d_model, tf.float32))
        div_term = 1.0 / div_term  # (d_model/2,)
        
        # Compute angles
        angles = tf.matmul(position, tf.expand_dims(div_term, axis=0))  # (max_len, d_model/2)
        
        # Create encoding
        pe = tf.zeros([max_len, d_model], dtype=tf.float32)
        
        # Even indices: sin
        pe_sin = tf.sin(angles)
        # Odd indices: cos
        pe_cos = tf.cos(angles)
        
        # Interleave sin and cos values
        indices = tf.range(0, d_model, delta=2)
        updates = tf.reshape(pe_sin, [-1])
        pe = tf.tensor_scatter_nd_update(
            pe, 
            tf.stack([
                tf.repeat(tf.range(max_len), tf.shape(indices)[0]),
                tf.tile(indices, [max_len])
            ], axis=1),
            updates
        )
        
        indices = tf.range(1, d_model, delta=2)
        updates = tf.reshape(pe_cos, [-1])
        pe = tf.tensor_scatter_nd_update(
            pe, 
            tf.stack([
                tf.repeat(tf.range(max_len), tf.shape(indices)[0]),
                tf.tile(indices, [max_len])
            ], axis=1),
            updates
        )
        
        # Add batch dimension and slice to required length
        pe = tf.expand_dims(pe[:seq_len], axis=0)
        return pe
    
    # For concrete values, use the simpler NumPy implementation
    positions = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(positions * div_term)
    pos_encoding[:, 1::2] = np.cos(positions * div_term)
    
    return tf.cast(tf.expand_dims(pos_encoding, 0), dtype=tf.float32)



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
    """
    Create an enhanced Audio Spectrogram Transformer model optimized for V100 GPU
    """
    if config is None:
        config = {}
    
    transformer_config = config.get("transformer", {})
    dim_feedforward = transformer_config.get("dim_feedforward", 512)  # Increased from 256
    dropout_rate = config.get("dropout", 0.3)
    l2_reg = config.get("l2_regularization", 1e-5)

    
    # Input layer expects (freq, time) format
    inputs = Input(shape=input_shape)
    
    # Reshape to prepare for CNN (batch, freq, time, 1)
    x = Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    # First convolutional block with L2 regularization - increased filters for V100
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', 
                              kernel_regularizer=l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(dropout_rate/2)(x)
    
    # Second convolutional block - increased filters
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                              kernel_regularizer=l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(dropout_rate/2)(x)
    
    # Third convolutional block - increased filters
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                              kernel_regularizer=l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(dropout_rate/2)(x)
    
    # Fourth convolutional block - new for V100
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same',
                              kernel_regularizer=l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dropout(dropout_rate/2)(x)

    
    # Global pooling instead of flattening to reduce parameters
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with batch normalization - increased dimensions for V100
    x = Dense(dim_feedforward, kernel_regularizer=l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    # Additional dense layer for V100
    x = Dense(dim_feedforward // 2, kernel_regularizer=l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    # Final classification layer
    x = Dense(64, kernel_regularizer=l2(l2_reg))(x)  # Increased from 32
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)

    # Use a threshold-adjusted sigmoid for better sensitivity to abnormal class
    outputs = Dense(1, activation="sigmoid", kernel_regularizer=l2(l2_reg), bias_initializer=tf.keras.initializers.Constant(-1.0))(x)
    outputs = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(outputs)
    
    return Model(inputs=inputs, outputs=outputs, name="AudioSpectrogramTransformer")


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
    Balance the dataset by augmenting the minority class
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
    
    # Create augmented samples
    augmented_data = []
    augmented_labels = []
    
    # Simple augmentation: add noise and small shifts
    for _ in range(n_to_add):
        # Randomly select a minority sample
        idx = np.random.choice(minority_indices)
        sample = train_data[idx].copy()
        
        # Add random noise
        noise_level = 0.1
        noise = np.random.normal(0, noise_level, sample.shape)
        augmented_sample = sample + noise
        
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



########################################################################
# main
########################################################################
def main():
    # Set memory growth before any other TensorFlow operations
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, False)
                logger.info(f"Using dynamic memory allocation for {device}")
            except Exception as e:
                logger.warning(f"Could not configure memory allocation for {device}: {e}")
    
        # Configure GPU memory for V100 32GB
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Allow TensorFlow to use most of the V100's 32GB memory
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 28)]  # 28GB limit (leaving some headroom)
                )
            logger.info("GPU memory limit set to 28GB for V100")
        except RuntimeError as e:
            logger.error(f"Error setting GPU memory limit: {e}")


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

    condition = parts[0]  # normal or abnormal
    db = parts[1]
    machine_type = parts[2]
    machine_id = parts[3].split('-')[0] if '-' in parts[3] else parts[3]
    print(f"DEBUG: Extracted - condition: {condition}, db: {db}, machine_type: {machine_type}, machine_id: {machine_id}")
    
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
    target_shape = (param["feature"]["n_mels"], 48)
    logger.info(f"Target spectrogram shape: {target_shape}")

    # Preprocess to ensure consistent shapes
    logger.info("Preprocessing training data...")
    train_data = preprocess_spectrograms(train_data, target_shape)
    logger.info(f"Preprocessed train data shape: {train_data.shape}")

    logger.info("Preprocessing validation data...")
    val_data = preprocess_spectrograms(val_data, target_shape)
    logger.info(f"Preprocessed validation data shape: {val_data.shape}")

    # Balance the dataset
    logger.info("Balancing dataset...")
    train_data, train_labels_expanded = balance_dataset(train_data, train_labels_expanded, augment_minority=True)

    abnormal_indices = np.where(train_labels_expanded == 1)[0]
    if len(abnormal_indices) > 0:
        logger.info(f"Adding extra augmentation for {len(abnormal_indices)} abnormal samples")
        
        # Create copies with different noise patterns
        augmented_abnormal = []
        for idx in abnormal_indices:
            # Create 3 variations of each abnormal sample
            for i in range(3):
                sample = train_data[idx].copy()
                # Add stronger noise to make variations more distinct
                noise_level = 0.15 + (i * 0.05)  # Increasing noise levels
                noise = np.random.normal(0, noise_level, sample.shape)
                augmented_sample = sample + noise
                augmented_sample = np.clip(augmented_sample, 0, 1)
                augmented_abnormal.append(augmented_sample)
        
        # Add the augmented samples to the training data
        if augmented_abnormal:
            augmented_abnormal = np.array(augmented_abnormal)
            train_data = np.vstack([train_data, augmented_abnormal])
            # Add corresponding labels (all 1 for abnormal)
            train_labels_expanded = np.concatenate([
                train_labels_expanded, 
                np.ones(len(augmented_abnormal))
            ])
            
            # Shuffle the combined dataset
            shuffle_indices = np.random.permutation(len(train_data))
            train_data = train_data[shuffle_indices]
            train_labels_expanded = train_labels_expanded[shuffle_indices]
            
            logger.info(f"After abnormal augmentation: {len(train_data)} samples, {np.sum(train_labels_expanded == 1)} abnormal")

    # Configure mixed precision
    mixed_precision_enabled = configure_mixed_precision(
        enabled=param.get("training", {}).get("mixed_precision", False)
    )

    monitor_gpu_usage()

    # Create model with the correct input shape
    model = create_ast_model(
        input_shape=(target_shape[0], target_shape[1]),
        config=param.get("model", {}).get("architecture", {})
    )

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

    mixed_precision_enabled = configure_mixed_precision(
        enabled=param.get("training", {}).get("mixed_precision", False)
    )


    batch_size = param.get("fit", {}).get("batch_size", 32)
    epochs = param.get("fit", {}).get("epochs", 100)
    base_learning_rate = param.get("fit", {}).get("compile", {}).get("learning_rate", 0.001)

    # Scale learning rate based on batch size
    learning_rate = get_scaled_learning_rate(base_learning_rate, batch_size)
    logger.info(f"Scaled learning rate from {base_learning_rate} to {learning_rate} for batch size {batch_size}")


    # Log the training parameters being used
    logger.info(f"Training with batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate}")

    if param.get("fit", {}).get("class_weight_balancing", True):
        # Count class occurrences
        class_counts = np.bincount(train_labels_expanded.astype(int))
        total_samples = np.sum(class_counts)
        
        # Calculate weights inversely proportional to class frequencies
        # But ensure abnormal class (1) gets higher weight
        abnormal_weight_multiplier = param.get("fit", {}).get("abnormal_weight_multiplier", 1.5)
        
        class_weights = {
            0: total_samples / (class_counts[0] * 2) if class_counts[0] > 0 else 1.0,
            1: (total_samples / (class_counts[1] * 2) * abnormal_weight_multiplier) 
            if len(class_counts) > 1 and class_counts[1] > 0 else 5.0
        }
        
        logger.info(f"Using calculated class weights: {class_weights}")
    else:
        # Use default weights that prioritize abnormal class
        class_weights = {
            0: 1.0,
            1: param.get("fit", {}).get("default_abnormal_weight", 5.0)
        }
        logger.info(f"Using default class weights: {class_weights}")



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

    model_config = param.get("model", {}).get("architecture", {})
    model.summary()

    if os.path.exists(model_file):
        model = tf.keras.models.load_model(model_file, custom_objects={"binary_cross_entropy_loss": binary_cross_entropy_loss})
        logger.info("Model loaded from file, no training performed")
    else:
        # Define callbacks
        callbacks = []
        
        # Early stopping
        early_stopping_config = param.get("fit", {}).get("early_stopping", {})
        if early_stopping_config.get("enabled", False):
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_config.get("monitor", "val_loss"),
                patience=20,  # Increase patience to allow more training
                min_delta=early_stopping_config.get("min_delta", 0.001),
                restore_best_weights=True
            ))

        callbacks.append(TerminateOnNaN(patience=3))
        logger.info("Added NaN detection callback")
                
        # Reduce learning rate on plateau
        lr_config = param.get("fit", {}).get("lr_scheduler", {})
        if lr_config.get("enabled", False):
            callbacks.append(ReduceLROnPlateau(
                monitor=lr_config.get("monitor", "val_loss"),
                factor=lr_config.get("factor", 0.1),
                patience=lr_config.get("patience", 5),
                min_delta=lr_config.get("min_delta", 0.001),
                cooldown=lr_config.get("cooldown", 2),
                min_lr=lr_config.get("min_lr", 0.00000001),
                verbose=1
            ))
        
        # Model checkpoint
        checkpoint_config = param.get("fit", {}).get("checkpointing", {})
        if checkpoint_config.get("enabled", False):
            checkpoint_path = f"{param['model_directory']}/checkpoint_ast.keras"
            callbacks.append(ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=checkpoint_config.get("monitor", "val_accuracy"),
                mode=checkpoint_config.get("mode", "max"),
                save_best_only=checkpoint_config.get("save_best_only", True),
                verbose=1
            ))
        
        # Add warmup learning rate scheduler if enabled
        warmup_config = param.get("fit", {}).get("warmup", {})
        if warmup_config.get("enabled", False):
            # Calculate total steps
            steps_per_epoch = len(train_data) // param["fit"]["batch_size"]
            total_steps = steps_per_epoch * param["fit"]["epochs"]
            
            # Calculate warmup steps
            warmup_epochs = warmup_config.get("epochs", 5)
            warmup_steps = steps_per_epoch * warmup_epochs
            
            # Create scheduler
            warmup_lr = WarmUpCosineDecayScheduler(
                learning_rate_base=learning_rate,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                hold_base_rate_steps=steps_per_epoch * warmup_config.get("hold_epochs", 0)
            )
            
            callbacks.append(warmup_lr)
            logger.info(f"Added warmup learning rate scheduler with {warmup_epochs} warmup epochs")

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
            compile_params["optimizer"] = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
            logger.info(f"Using Adam optimizer with gradient clipping (clipnorm={clipnorm})")


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
                compile_params["loss"] = lambda y_true, y_pred: focal_loss(y_true, y_pred, gamma, alpha)
            else:
                logger.warning("Using standard focal loss - watch for NaN losses")
                compile_params["loss"] = tf.keras.losses.BinaryFocalCrossentropy(
                    gamma=gamma, alpha=alpha, from_logits=False
                )
        else:
            logger.info("Using standard binary crossentropy loss")
            compile_params["loss"] = "binary_crossentropy"

        
        compile_params["metrics"] = ['accuracy']
        model.compile(**compile_params)
        
        # Sample weights for class balance
        sample_weights = np.ones(len(train_data))
        
        # Apply uniform sample weights - no machine-specific weighting
        if param.get("fit", {}).get("apply_sample_weights", False):
            weight_factor = param.get("fit", {}).get("weight_factor", 1.0)
            sample_weights *= weight_factor
        
        if param.get("training", {}).get("gradient_accumulation_steps", 1) > 1:
            logger.info(f"Using gradient accumulation with {param['training']['gradient_accumulation_steps']} steps")
            
            # Create optimizer
            optimizer = compile_params["optimizer"]
            loss_fn = compile_params["loss"]
            
            # Get the dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels_expanded))
            train_dataset = train_dataset.batch(param["fit"]["batch_size"])
            
            # Create validation dataset
            val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels_expanded))
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
                    
                    # Reshape y_batch to match logits shape
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
                
                # Calculate accuracy - reshape y_batch to match logits
                y_pred = tf.cast(tf.greater_equal(logits, 0.5), tf.float32)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(y_batch_reshaped, y_pred), tf.float32))
                
                return loss_value, accuracy



            
            # Define validation step function
            @tf.function
            def val_step(x_batch, y_batch):
                logits = model(x_batch, training=False)
                
                # Reshape y_batch to match logits shape
                y_batch_reshaped = tf.reshape(y_batch, logits.shape)
                
                if isinstance(loss_fn, str):
                    if loss_fn == "binary_crossentropy":
                        loss_value = tf.keras.losses.binary_crossentropy(y_batch_reshaped, logits)
                    else:
                        loss_value = tf.keras.losses.get(loss_fn)(y_batch_reshaped, logits)
                else:
                    loss_value = loss_fn(y_batch_reshaped, logits)
                
                # Calculate accuracy
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
                            # Save best model
                            model.save(model_file)
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
                            # Save best model
                            model.save(model_file)
                            logger.info(f"Saved best model at epoch {epoch+1}")
                        else:
                            early_stopping_callback.wait += 1
                            if early_stopping_callback.wait >= early_stopping_callback.patience:
                                print(f"Early stopping triggered at epoch {epoch+1}")
                                break
                
                # Save model periodically
                if (epoch + 1) % 5 == 0:
                    model.save(f"{param['model_directory']}/model_overall_ast_epoch_{epoch+1}.keras")
                    logger.info(f"Saved model checkpoint at epoch {epoch+1}")
                
                # Clear memory between epochs
                gc.collect()
            
            # Convert history to a format compatible with Keras history
            history = type('History', (), {'history': history})


        else:
            if param.get("training", {}).get("xla_acceleration", False):
                logger.info("Enabling XLA acceleration")
                tf.config.optimizer.set_jit(True)
                os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
            
            # Check for data shape mismatch and fix it
            if train_data.shape[0] != train_labels_expanded.shape[0]:
                logger.warning(f"Data shape mismatch! X: {train_data.shape[0]} samples, y: {train_labels_expanded.shape[0]} labels")
                
                if train_data.shape[0] > train_labels_expanded.shape[0]:
                    # Too many features, need to reduce
                    train_data = train_data[:train_labels_expanded.shape[0]]
                    sample_weights = sample_weights[:train_labels_expanded.shape[0]]
                    logger.info(f"Reduced X to match y: {train_data.shape}")
                else:
                    #Too many labels, need to reduce
                    train_labels_expanded = train_labels_expanded[:train_data.shape[0]]
                    sample_weights = sample_weights[:train_data.shape[0]]
                    logger.info(f"Reduced y to match X: {train_labels_expanded.shape}")

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


            # Use a smaller batch size and more conservative approach
            safe_batch_size = min(64, batch_size)
            safe_epochs = min(100, epochs)

            logger.info(f"Training with conservative settings: batch_size={safe_batch_size}, epochs={safe_epochs}")
            history = model.fit(
                train_data,
                train_labels_expanded,
                batch_size=safe_batch_size,
                epochs=safe_epochs,
                validation_data=(val_data, val_labels_expanded),
                callbacks=callbacks,
                class_weight=class_weights,
                sample_weight=sample_weights,
                verbose=1
            )


            # Save the trained model
            model.save(model_file)
            logger.info(f"Model saved to {model_file}")

        # Visualize training history
        visualizer.loss_plot(history, machine_type=machine_type, machine_id=machine_id, db=db)
        visualizer.save_figure(history_img)

    # Log training time
    training_time = time.time() - model_start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")

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

    # Evaluate the model
    if test_data.shape[0] > 0:
        # Predict on test set
        y_pred = model.predict(test_data, batch_size=batch_size, verbose=1)
        # Use a lower threshold to increase sensitivity to abnormal class
        detection_threshold = 0.3  # Lower threshold to catch more abnormal samples
        y_pred_binary = (y_pred > detection_threshold).astype(int)
        logger.info(f"Using adjusted detection threshold: {detection_threshold}")
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
        results["timing"] = {
            "total_execution_time_seconds": float(total_time),
            "model_training_time_seconds": float(training_time),
        }

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

    # Save results to YAML
    with open(result_file, "w") as f:
        yaml.safe_dump(results, f)
    logger.info(f"Results saved to {result_file}")

    # Log total execution time
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()