#!/usr/bin/env python
"""
 @file   baseline_fnn.py
 @brief  Baseline code of simple AE-based anomaly detection used experiment in [1], updated for 2025 with enhancements.
 @author Ryo Tanabe and Yohei Kawaguchi (Hitachi Ltd.), updated by Lamnaouar Ayoub (Github: lamnaouarrr), further modified by Grok
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
from tensorflow.keras import mixed_precision
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, Add, MultiHeadAttention, LayerNormalization, Reshape, Permute, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.losses import mean_squared_error
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
__versions__ = "2.1.0"
########################################################################

def binary_cross_entropy_loss(y_true, y_pred):
    """
    Binary cross-entropy loss for autoencoder
    """
    # Normalize inputs if needed (values between 0 and 1)
    y_true_normalized = (y_true - K.min(y_true)) / (K.max(y_true) - K.min(y_true) + K.epsilon())
    y_pred_normalized = (y_pred - K.min(y_pred)) / (K.max(y_pred) - K.min(y_pred) + K.epsilon())
    
    return tf.keras.losses.binary_crossentropy(y_true_normalized, y_pred_normalized)

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
    os.makedirs("./logs/log_fnn", exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, filename="./logs/log_fnn/baseline_fnn.log")
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
            
        if y is None:
            logger.error(f"Failed to load {file_name}")
            return None
        
        # Skip files that are too short
        if len(y) < n_fft:
            logger.warning(f"File too short: {file_name}")
            return None
            
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
        
        # Ensure minimum size for transformer input
        min_time_dim = param.get("model", {}).get("architecture", {}).get("transformer", {}).get("patch_size", 16) * 4
        if log_mel_spectrogram.shape[1] < min_time_dim:
            # Pad if too short
            pad_width = min_time_dim - log_mel_spectrogram.shape[1]
            log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        
        # Trim or sample if too long
        max_time_dim = 512  # Maximum time dimension
        if log_mel_spectrogram.shape[1] > max_time_dim:
            start = np.random.randint(0, log_mel_spectrogram.shape[1] - max_time_dim)
            log_mel_spectrogram = log_mel_spectrogram[:, start:start+max_time_dim]
            
        return log_mel_spectrogram
        
    except Exception as e:
        logger.error(f"Error in file_to_spectrogram for {file_name}: {e}")
        return None

def list_to_spectrograms(file_list, labels=None, msg="calc...", augment=False, param=None, batch_size=20):
    """
    Process a list of files into spectrograms with optional labels
    
    Args:
        file_list: List of audio file paths
        labels: Optional labels for each file
        msg: Progress message
        augment: Whether to apply augmentation
        param: Parameters dictionary
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (spectrograms, labels) if labels provided, otherwise just spectrograms
    """
    n_mels = param.get("feature", {}).get("n_mels", 64)
    n_fft = param.get("feature", {}).get("n_fft", 1024)
    hop_length = param.get("feature", {}).get("hop_length", 512)
    power = param.get("feature", {}).get("power", 2.0)
    
    spectrograms = []
    processed_labels = []
    frames = param.get("feature", {}).get("frames", None)
    is_frame_based = frames is not None and frames > 0
    
    # Process files in batches to manage memory
    for batch_start in tqdm(range(0, len(file_list), batch_size), desc=f"{msg} (in batches)"):
        batch_end = min(batch_start + batch_size, len(file_list))
        batch_files = file_list[batch_start:batch_end]
        
        if labels is not None:
            batch_labels = labels[batch_start:batch_end]
        
        for idx, file_path in enumerate(batch_files):
            try:
                spectrogram = file_to_spectrogram(file_path,
                                              n_mels=n_mels,
                                              n_fft=n_fft,
                                              hop_length=hop_length,
                                              power=power,
                                              augment=augment,
                                              param=param)
                
                if spectrogram is not None:
                    if is_frame_based and len(spectrogram.shape) == 3:
                        spectrograms.append(spectrogram[0])
                        if labels is not None:
                            processed_labels.append(batch_labels[idx])
                    else:
                        spectrograms.append(spectrogram)
                        if labels is not None:
                            processed_labels.append(batch_labels[idx])
                        
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
    
    # Convert to numpy arrays
    if not spectrograms:
        if labels is not None:
            return np.array([]), np.array([])
        return np.array([])
    
    # Determine max dimensions
    max_freq = max(spec.shape[0] for spec in spectrograms)
    max_time = max(spec.shape[1] for spec in spectrograms)
    
    # Create output array with consistent dimensions
    batch_spectrograms = np.zeros((len(spectrograms), max_freq, max_time), dtype=np.float32)
    
    # Fill the array
    for i, spec in enumerate(spectrograms):
        batch_spectrograms[i, :spec.shape[0], :spec.shape[1]] = spec
    
    if labels is not None:
        return batch_spectrograms, np.array(processed_labels)
    return batch_spectrograms




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

########################################################################
# model
########################################################################
def create_ast_model(input_shape, config=None):
    """
    Create an Audio Spectrogram Transformer model
    
    Args:
        input_shape: Shape of input spectrograms (freq, time)
        config: Model configuration parameters
        
    Returns:
        Keras Model instance
    """
    if config is None:
        config = {}
    
    transformer_config = config.get("transformer", {})
    num_heads = transformer_config.get("num_heads", 8)
    dim_feedforward = transformer_config.get("dim_feedforward", 512)
    num_encoder_layers = transformer_config.get("num_encoder_layers", 4)
    dropout_rate = config.get("dropout", 0.1)
    use_positional_encoding = transformer_config.get("positional_encoding", True)
    use_pretrained = transformer_config.get("use_pretrained", False)
    patch_size = transformer_config.get("patch_size", 16)
    attention_dropout = transformer_config.get("attention_dropout", 0.1)
    attention_type = transformer_config.get("attention_type", "standard")
    key_dim = dim_feedforward // num_heads
    pos_encoding_type = transformer_config.get("pos_encoding_type", "sinusoidal")
    layer_norm_epsilon = float(transformer_config.get("layer_norm_epsilon", 1e-6))
    activation_fn = transformer_config.get("activation_fn", "gelu")
    ff_dim_multiplier = transformer_config.get("ff_dim_multiplier", 4)
    enable_rotary = transformer_config.get("enable_rotary", False)
    
    # Input layer expects (freq, time) format
    inputs = Input(shape=input_shape)
    
    # Reshape to prepare for transformer (batch, freq, time, 1)
    x = Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    if use_pretrained and K.image_data_format() == 'channels_last':
        # Using pre-trained ViT model
        try:
            # Create configuration for the ViT model
            vit_config = ViTConfig(
                hidden_size=768,
                num_hidden_layers=num_encoder_layers,
                num_attention_heads=num_heads,
                intermediate_size=dim_feedforward,
                hidden_dropout_prob=dropout_rate,
                attention_probs_dropout_prob=attention_dropout,
                image_size=224,  # Standard for ViT
                patch_size=patch_size
            )
            
            # Resize input to fit pre-trained model
            # This needs a fixed size input - adjust based on your specific needs
            x = tf.keras.layers.experimental.preprocessing.Resizing(224, 224)(x)
            
            # Pre-trained model - you'll need to install transformers
            vit_model = TFViTModel.from_pretrained(
                transformer_config.get("pretrained_model", "google/vit-base-patch16-224"),
                config=vit_config
            )
            
            # Get the output from the ViT model
            vit_output = vit_model(x).last_hidden_state
            
            # Global average pooling over the sequence dimension
            x = GlobalAveragePooling1D()(vit_output)
            
        except Exception as e:
            logger.warning(f"Failed to load pre-trained ViT model: {e}. Falling back to custom transformer.")
            use_pretrained = False
    
    if not use_pretrained:
        # Custom transformer implementation
        # Extract patches from the spectrogram
        freq_patches = input_shape[0] // patch_size
        time_patches = input_shape[1] // patch_size
        total_patches = freq_patches * time_patches
        patch_dim = patch_size * patch_size
        
        # Reshape into patches
        x = tf.keras.layers.Conv2D(
            filters=dim_feedforward,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid"
        )(x)

        # Get the actual dimensions after convolution
        conv_shape = tf.shape(x)
        height, width = conv_shape[1], conv_shape[2]
        total_patches = height * width

        # Flatten patches to sequence - dynamic reshape based on actual dimensions
        x = tf.reshape(x, [-1, total_patches, dim_feedforward])
        
        # Add positional encoding
        if use_positional_encoding:
            if pos_encoding_type == "rotary" and enable_rotary:
                # Rotary embeddings are applied within attention calculation
                # This is a flag to enable it, actual implementation would be in the attention mechanism
                pass
            else:
                # Get a concrete value for seq_len based on the shape tensor
                seq_len = tf.shape(x)[1]
                pos_encoding = positional_encoding(seq_len, dim_feedforward, pos_encoding_type)
                
                # Cast positional encoding to match the data type of x
                pos_encoding = tf.cast(pos_encoding, dtype=x.dtype)
                
                if pos_encoding_type != "alibi":  # alibi is used directly in attention
                    x = x + pos_encoding


        
        # Transformer encoder blocks
        for _ in range(num_encoder_layers):
            # Layer normalization
            attn_input = LayerNormalization(epsilon=float(layer_norm_epsilon))(x)

            
            # Multi-head attention
            if attention_type == "standard" or attention_type == "efficient":
                # Use standard MultiHeadAttention for both types
                attn_output = MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=key_dim,
                    dropout=attention_dropout
                )(attn_input, attn_input)
            elif attention_type == "linear":
                # Linear attention implementation (approximate attention)
                query = Dense(dim_feedforward)(attn_input)
                key = Dense(dim_feedforward)(attn_input)
                value = Dense(dim_feedforward)(attn_input)
                
                # Reshape for heads
                batch_size = tf.shape(query)[0]
                seq_length = tf.shape(query)[1]
                query = tf.reshape(query, [batch_size, seq_length, num_heads, key_dim])
                key = tf.reshape(key, [batch_size, seq_length, num_heads, key_dim])
                value = tf.reshape(value, [batch_size, seq_length, num_heads, key_dim])
                
                # Linear attention calculation (ϕ(q)·ϕ(k)ᵀ·v)
                query = tf.nn.elu(query) + 1.0
                key = tf.nn.elu(key) + 1.0
                
                # Compute attention
                kv = tf.einsum("bshd,bshv->bhdv", key, value)
                attn_output = tf.einsum("bshd,bhdv->bshv", query, kv)
                
                # Reshape back
                attn_output = tf.reshape(attn_output, [batch_size, seq_length, dim_feedforward])
                attn_output = Dense(dim_feedforward)(attn_output)
            else:
                # Default to standard attention
                attn_output = MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=key_dim,
                    dropout=attention_dropout
                )(attn_input, attn_input)
            
            # Skip connection
            x = x + attn_output
            
            # Feed-forward network
            ffn_input = LayerNormalization(epsilon=float(layer_norm_epsilon))(x)
            if activation_fn == "gelu":
                ffn_output = Dense(dim_feedforward * ff_dim_multiplier, activation="gelu")(ffn_input)
            elif activation_fn == "relu":
                ffn_output = Dense(dim_feedforward * ff_dim_multiplier, activation="relu")(ffn_input)
            elif activation_fn == "swish":
                ffn_output = Dense(dim_feedforward * ff_dim_multiplier)(ffn_input)
                ffn_output = tf.keras.activations.swish(ffn_output)
            else:
                # Default to gelu
                ffn_output = Dense(dim_feedforward * ff_dim_multiplier, activation="gelu")(ffn_input)
            ffn_output = Dense(dim_feedforward)(ffn_output)
            
            if dropout_rate > 0:
                ffn_output = Dropout(dropout_rate)(ffn_output)
                
            # Skip connection
            x = x + ffn_output
        
        # Final layer normalization
        x = LayerNormalization(epsilon=float(1e-6))(x)
        
        # Global average pooling over sequence dimension
        x = GlobalAveragePooling1D()(x)
    
    # Output layers
    x = Dense(dim_feedforward // 2, activation="gelu")(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    
    return Model(inputs=inputs, outputs=outputs, name="AudioSpectrogramTransformer")


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


########################################################################
# main
########################################################################
def main():

    # Configure GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                logger.info(f"Memory growth enabled for {device}")
            except Exception as e:
                logger.warning(f"Could not set memory growth for {device}: {e}")


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
    result_file = f"{param['result_directory']}/result_fnn.yaml"

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

    train_data, train_labels_expanded = list_to_spectrograms(
        train_files,
        train_labels,
        msg="generate train_dataset",
        augment=True,
        param=param,
        batch_size=20
    )
    print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels_expanded.shape}")

    val_data, val_labels_expanded = list_to_spectrograms(
        val_files,
        val_labels,
        msg="generate validation_dataset",
        augment=False,
        param=param,
        batch_size=20
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

    #debug
    normal_count = sum(1 for label in train_labels_expanded if label == 0)
    abnormal_count = sum(1 for label in train_labels_expanded if label == 1)
    print(f"Training data composition: Normal={normal_count}, Abnormal={abnormal_count}")

    
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
    model = create_ast_model(
        input_shape=(train_data.shape[1], train_data.shape[2]),
        config=model_config
    )
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
                patience=early_stopping_config.get("patience", 10),
                min_delta=early_stopping_config.get("min_delta", 0.001),
                restore_best_weights=early_stopping_config.get("restore_best_weights", True)
            ))
        
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
        
        compile_params = param["fit"]["compile"].copy()
        loss_type = param.get("model", {}).get("loss", "binary_crossentropy")
        
        # Handle learning_rate separately for the optimizer
        learning_rate = compile_params.pop("learning_rate", 0.0001)
        
        if loss_type == "binary_crossentropy":
            compile_params["loss"] = binary_cross_entropy_loss
        else:
            compile_params["loss"] = "binary_crossentropy"
        
        if "optimizer" in compile_params and compile_params["optimizer"] == "adam":
            compile_params["optimizer"] = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        compile_params["metrics"] = ['accuracy']
        model.compile(**compile_params)
        
        # Sample weights for class balance
        sample_weights = np.ones(len(train_data))
        
        # Apply uniform sample weights - no machine-specific weighting
        if param.get("fit", {}).get("apply_sample_weights", False):
            weight_factor = param.get("fit", {}).get("weight_factor", 1.0)
            sample_weights *= weight_factor
        
        if False and param.get("training", {}).get("gradient_accumulation_steps", 1) > 1:
            logger.info(f"Using manual gradient accumulation with {param['training']['gradient_accumulation_steps']} steps")
            
            # Create optimizer without accumulation
            optimizer = compile_params["optimizer"]
            
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
            
            # Create accumulated gradients outside the function
            accumulated_grads = None

            def initialize_accumulated_grads(model):
                """Initialize accumulated gradients with zeros for each trainable variable"""
                global accumulated_grads
                accumulated_grads = [tf.Variable(tf.zeros_like(var)) for var in model.trainable_variables]

            # Initialize accumulated gradients
            initialize_accumulated_grads(model)


            # Manual training loop
            @tf.function
            def train_step(x_batch, y_batch, step):
                global accumulated_grads
                
                with tf.GradientTape() as tape:
                    logits = model(x_batch, training=True)
                    loss_value = compile_params["loss"](y_batch, logits)
                    loss_value = loss_value / tf.cast(accum_steps, dtype=loss_value.dtype)
                    
                # Accumulate gradient
                grads = tape.gradient(loss_value, model.trainable_variables)
                
                for i, g in enumerate(grads):
                    accumulated_grads[i].assign_add(g)
                    
                # If we've accumulated enough gradients, apply them and reset
                if (step + 1) % accum_steps == 0:
                    optimizer.apply_gradients(zip(accumulated_grads, model.trainable_variables))
                    for i in range(len(accumulated_grads)):
                        accumulated_grads[i].assign(tf.zeros_like(accumulated_grads[i]))
                        
                # Calculate accuracy - ensure consistent data types
                # Convert both to the same data type (float32)
                y_batch_float32 = tf.cast(y_batch, tf.float32)
                y_pred = tf.cast(tf.greater_equal(logits, 0.5), tf.float32)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(y_batch_float32, y_pred), tf.float32))
                        
                return loss_value, accuracy


            
            @tf.function
            def val_step(x_batch, y_batch):
                logits = model(x_batch, training=False)
                loss_value = compile_params["loss"](y_batch, logits)
                
                # Calculate accuracy - ensure consistent data types
                y_batch_float32 = tf.cast(y_batch, tf.float32)
                y_pred = tf.cast(tf.greater_equal(logits, 0.5), tf.float32)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(y_batch_float32, y_pred), tf.float32))
                
                return loss_value, accuracy

            
            # Training loop
            epochs = param["fit"]["epochs"]
            for epoch in range(epochs):
                print(f"\nEpoch {epoch+1}/{epochs}")
                
                # Reset accumulated gradients at the beginning of each epoch
                for i in range(len(accumulated_grads)):
                    accumulated_grads[i].assign(tf.zeros_like(accumulated_grads[i]))
                
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
                    batch_loss, batch_accuracy = train_step(x_batch, y_batch, step % accum_steps)
                    train_loss.update_state(batch_loss)
                    train_accuracy.update_state(batch_accuracy)
                    step += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f'{train_loss.result():.4f}',
                        'accuracy': f'{train_accuracy.result():.4f}'
                    })
                
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
                    # Implement early stopping logic here if needed
                    pass
            
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
                    # Too many labels, need to reduce
                    train_labels_expanded = train_labels_expanded[:train_data.shape[0]]
                    logger.info(f"Reduced y to match X: {train_labels_expanded.shape}")

            max_samples = 50000

            if train_data.shape[0] > max_samples:
                indices = np.random.choice(train_data.shape[0], max_samples, replace=False)
                train_data_subset = train_data[indices]
                train_labels_subset = train_labels_expanded[indices]
                sample_weights_subset = sample_weights[indices]
                logger.info(f"Using subset of data: {train_data_subset.shape}")
            else:
                train_data_subset = train_data
                train_labels_subset = train_labels_expanded
                sample_weights_subset = sample_weights


            print(f"Final training shapes - X: {train_data.shape}, y: {train_labels.shape}, weights: {sample_weights.shape}")
            history = model.fit(
                train_data,
                train_labels_expanded,
                epochs=param["fit"]["epochs"],
                batch_size=param["fit"]["batch_size"],
                shuffle=param["fit"]["shuffle"],
                validation_data=(val_data, val_labels_expanded),
                verbose=param["fit"]["verbose"],
                callbacks=callbacks,
                sample_weight=sample_weights
            )
        
        model.save(model_file)
        visualizer.loss_plot(history)
        visualizer.save_figure(history_img)
    

    if not os.path.exists(model_file):
        # Capture the final training and validation accuracies
        train_accuracy = history.history['accuracy'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
        
        # Store these in the results dictionary
        evaluation_result["TrainAccuracy"] = float(train_accuracy)
        evaluation_result["ValidationAccuracy"] = float(val_accuracy)
        
        logger.info(f"Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
    else:
        # If model was loaded from file and not trained, we need to evaluate on train and validation data
        train_pred = model.predict(train_data, verbose=0)
        train_pred_binary = (train_pred.flatten() >= 0.5).astype(int)
        train_accuracy = metrics.accuracy_score(train_labels_expanded, train_pred_binary)
        
        val_pred = model.predict(val_data, verbose=0)
        val_pred_binary = (val_pred.flatten() >= 0.5).astype(int)
        val_accuracy = metrics.accuracy_score(val_labels_expanded, val_pred_binary)

        # Calculate model training time
        model_end_time = time.time()
        model_training_time = model_end_time - model_start_time
        logger.info(f"Model training time: {model_training_time:.2f} seconds")
        
        # Store these in the results dictionary
        evaluation_result["TrainAccuracy"] = float(train_accuracy)
        evaluation_result["ValidationAccuracy"] = float(val_accuracy)
        
        logger.info(f"Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")


    print("============== EVALUATION ==============")
    y_pred = []
    y_true = test_labels

    for num, file_name in tqdm(enumerate(test_files), total=len(test_files)):
        try:
            data = file_to_spectrogram(file_name,
                                n_mels=param["feature"]["n_mels"],
                                n_fft=param["feature"]["n_fft"],
                                hop_length=param["feature"]["hop_length"],
                                power=param["feature"]["power"],
                                augment=False,  # No augmentation during eval
                                param=param)
                                    
            if data is None:
                logger.warning(f"No valid features extracted from file: {file_name}")
                continue
            
            # Reshape for batch prediction
            data = np.expand_dims(data, axis=0)
                    
            # Get the predicted class probability
            pred = model.predict(data, verbose=0)
            # Use the single prediction (no need to average frames)
            file_pred = float(pred[0][0])
            y_pred.append(file_pred)
        except Exception as e:
            logger.warning(f"Error processing file: {file_name}, error: {e}")
            # If there's an error, use a default value (e.g., 0.5)
            default_prediction = param.get("dataset", {}).get("default_prediction", 0.5)
            y_pred.append(default_prediction)

    # Optimize threshold using ROC curve if enabled
    if param.get("model", {}).get("threshold_optimization", True) and len(y_true) > 0 and len(np.unique(y_true)) > 1:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        logger.info(f"Optimal threshold: {optimal_threshold:.6f}")
    else:
        optimal_threshold = 0.5  # Default threshold
        logger.info(f"Using default threshold: {optimal_threshold}")

    # Convert predictions to binary using optimal threshold
    y_pred_binary = (np.array(y_pred) >= optimal_threshold).astype(int)



    # Plot and save confusion matrix
    visualizer.plot_confusion_matrix(y_true, y_pred_binary)
    visualizer.save_figure(f"{param['result_directory']}/confusion_matrix_{evaluation_result_key}.png")


    #debug####################################################################
    print(f"DEBUG - y_true values distribution: {np.unique(y_true, return_counts=True)}")
    print(f"DEBUG - y_pred_binary values distribution: {np.unique(y_pred_binary, return_counts=True)}")
    print(f"DEBUG - First 10 pairs of true and predicted values:")
    for i in range(min(10, len(y_true))):
        print(f"  True: {y_true[i]}, Predicted: {y_pred_binary[i]}")

    if len(y_true) == 0:
        logger.error("Test set is empty! No samples to evaluate.")
        # Use placeholder values instead of generating a classification report
        class_report = {
            "0": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0},
            "1": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}
        }
    elif len(np.unique(y_pred_binary)) == 1:
        logger.warning(f"All predictions are the same class: {np.unique(y_pred_binary)[0]}! Classification metrics will be 0.")
        # Create a more detailed class report with actual counts
        classes = np.unique(np.concatenate([y_true, y_pred_binary]))
        class_report = {}
        for cls in classes:
            cls_support = np.sum(y_true == cls)
            tp = np.sum((y_true == cls) & (y_pred_binary == cls))
            precision = tp / np.sum(y_pred_binary == cls) if np.sum(y_pred_binary == cls) > 0 else 0
            recall = tp / cls_support if cls_support > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            class_report[str(cls)] = {
                "precision": float(precision), 
                "recall": float(recall), 
                "f1-score": float(f1), 
                "support": int(cls_support)
            }
        
        # Ensure both 0 and 1 classes are present
        for cls in [0, 1]:
            if str(cls) not in class_report:
                class_report[str(cls)] = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
    else:
        # Generate and print classification report
        class_report = classification_report(y_true, y_pred_binary, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_binary))


    ###########################################################################

    # Calculate accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred_binary)

    
    evaluation_result["TestAccuracy"] = float(accuracy)

    # Add this safety check function before accessing class_report
    def get_safe_metric(report, class_label, metric, default=0.0):
        """Safely retrieve a metric from classification report with a default fallback."""
        if str(class_label) in report:
            return float(report[str(class_label)][metric])
        return default

    # Then use it for all metrics
    evaluation_result["Precision"] = {
        "class_0": get_safe_metric(class_report, "0.0", "precision"),
        "class_1": get_safe_metric(class_report, "1.0", "precision")
    }
    evaluation_result["Recall"] = {
        "class_0": get_safe_metric(class_report, "0.0", "recall"),
        "class_1": get_safe_metric(class_report, "1.0", "recall")
    }
    evaluation_result["F1Score"] = {
        "class_0": get_safe_metric(class_report, "0.0", "f1-score"),
        "class_1": get_safe_metric(class_report, "1.0", "f1-score")
    }
    evaluation_result["Support"] = {
        "class_0": int(get_safe_metric(class_report, "0.0", "support")),
        "class_1": int(get_safe_metric(class_report, "1.0", "support"))
    }

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    results[evaluation_result_key] = evaluation_result


    #add the machine's predictions and true labels to the overall collection
    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred_binary)

    print("===========================")

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    results["execution_time_seconds"] = float(total_time)
    results["model_training_time_seconds"] = float(model_training_time) if 'model_training_time' in locals() else 0.0

    print("\n===========================")
    logger.info(f"all results -> {result_file}")
    with open(result_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print("===========================")

if __name__ == "__main__":
    main()