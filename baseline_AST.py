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
__versions__ = "3.0.0"
########################################################################

def binary_cross_entropy_loss(y_true, y_pred):
    """
    Binary cross-entropy loss for autoencoder
    """
    # Normalize inputs if needed (values between 0 and 1)
    y_true_normalized = (y_true - K.min(y_true)) / (K.max(y_true) - K.min(y_true) + K.epsilon())
    y_pred_normalized = (y_pred - K.min(y_pred)) / (K.max(y_pred) - K.min(y_pred) + K.epsilon())
    
    return tf.keras.losses.binary_crossentropy(y_true_normalized, y_pred_normalized)

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal loss for addressing class imbalance
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        gamma: Focusing parameter (higher values focus more on hard examples)
        alpha: Weighting factor for the positive class
        
    Returns:
        Focal loss value
    """
    # Clip predictions to prevent numerical instability
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate cross entropy
    cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
    
    # Apply weighting for positive and negative classes
    alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
    
    # Apply focusing parameter
    focal_weight = K.pow(1 - y_true * y_pred - (1 - y_true) * (1 - y_pred), gamma)
    
    # Combine all factors
    loss = alpha_weight * focal_weight * cross_entropy
    
    return K.mean(loss)

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
                    # CRITICAL FIX: Handle frame-based spectrograms properly
                    if is_frame_based and len(spectrogram.shape) == 3:
                        # Option 1: Only use the first frame to avoid label duplication
                        spectrograms.append(spectrogram[0])
                        if labels is not None:
                            processed_labels.append(batch_labels[idx])
                        
                        # Option 2 (commented out): Use all frames but duplicate labels
                        # for frame in spectrogram:
                        #     spectrograms.append(frame)
                        #     if labels is not None:
                        #         processed_labels.append(batch_labels[idx])
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


def configure_mixed_precision(enabled=True):
    """
    Configure mixed precision training with proper error handling.
    
    Args:
        enabled: Whether to enable mixed precision
        
    Returns:
        True if mixed precision was successfully enabled, False otherwise
    """
    if not enabled:
        logger.info("Mixed precision training disabled")
        return False
    
    try:
        # Check if GPU is available
        if not tf.config.list_physical_devices('GPU'):
            logger.warning("No GPU found, disabling mixed precision")
            return False
        
        # Check TensorFlow version
        import tensorflow as tf
        if tf.__version__.startswith('1.'):
            logger.warning("Mixed precision requires TensorFlow 2.x, disabling")
            return False
        
        # Import mixed precision module
        from tensorflow.keras import mixed_precision
        
        # Configure policy
        policy_name = 'mixed_float16'
        logger.info(f"Enabling mixed precision with policy: {policy_name}")
        mixed_precision.set_global_policy(policy_name)
        
        # Verify policy was set
        current_policy = mixed_precision.global_policy()
        logger.info(f"Mixed precision policy enabled: {current_policy}")
        
        # Check if policy was actually set to mixed_float16
        if str(current_policy) != policy_name:
            logger.warning(f"Failed to set mixed precision policy, got {current_policy}")
            return False
        
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

########################################################################
# model
########################################################################
def create_ast_model(input_shape, config=None):
    """
    Create a simpler Audio Spectrogram Transformer model with improved architecture
    """
    if config is None:
        config = {}
    
    transformer_config = config.get("transformer", {})
    num_heads = transformer_config.get("num_heads", 4)
    dim_feedforward = transformer_config.get("dim_feedforward", 256)
    num_encoder_layers = transformer_config.get("num_encoder_layers", 2)
    dropout_rate = config.get("dropout", 0.3)  # Increased dropout for better generalization
    
    # Input layer expects (freq, time) format
    inputs = Input(shape=input_shape)
    
    # Reshape to prepare for CNN (batch, freq, time, 1)
    x = Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    # First convolutional block
    x = tf.keras.layers.Conv2D(32, (5, 5), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(dropout_rate/2)(x)
    
    # Second convolutional block
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(dropout_rate/2)(x)
    
    # Third convolutional block
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate/2)(x)
    
    # Fourth convolutional block
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Global pooling instead of flattening to reduce parameters
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with batch normalization
    x = Dense(dim_feedforward)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(dim_feedforward // 2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(1, activation="sigmoid")(x)
    
    return Model(inputs=inputs, outputs=outputs, name="AudioSpectrogramTransformer")




# Replace the current preprocess_spectrograms function with this:
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
    Applies mixup augmentation to the data
    
    Args:
        x: Input features
        y: Target labels
        alpha: Mixup interpolation strength
        
    Returns:
        Mixed inputs, mixed targets, and lambda value
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
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

def configure_mixed_precision(enabled=True):
    """
    Configure mixed precision training with proper error handling.
    
    Args:
        enabled: Whether to enable mixed precision
        
    Returns:
        True if mixed precision was successfully enabled, False otherwise
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
        
        # Configure policy
        policy_name = 'mixed_float16'
        logger.info(f"Enabling mixed precision with policy: {policy_name}")
        mixed_precision.set_global_policy(policy_name)
        
        # Verify policy was set
        current_policy = mixed_precision.global_policy()
        logger.info(f"Mixed precision policy enabled: {current_policy}")
        
        # Check if policy was actually set to mixed_float16
        if str(current_policy) != policy_name:
            logger.warning(f"Failed to set mixed precision policy, got {current_policy}")
            return False
        
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


# Add this function after the preprocess_spectrograms function
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

    # Define target shape for spectrograms
    target_shape = (param["feature"]["n_mels"], 64)  # Adjust time dimension as needed
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

    # Configure mixed precision
    mixed_precision_enabled = configure_mixed_precision(
        enabled=param.get("training", {}).get("mixed_precision", False)
    )

    # Create model with the correct input shape
    model = create_ast_model(
        input_shape=(target_shape[0], target_shape[1]),
        config=param.get("model", {}).get("architecture", {})
    )

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
    learning_rate = param.get("fit", {}).get("compile", {}).get("learning_rate", 0.001)

    # Log the training parameters being used
    logger.info(f"Training with batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate}")

    # For class weights, check if they're already being calculated in your existing code
    # If you want to ensure abnormal class gets higher weight, you can adjust the calculation:
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
        if mixed_precision_enabled and compile_params.get("optimizer") == "adam":
            from tensorflow.keras.optimizers import Adam
            
            # In TF 2.4+, LossScaleOptimizer is automatically applied when using mixed_float16 policy
            # So we just need to create the base optimizer
            compile_params["optimizer"] = Adam(learning_rate=learning_rate)
            logger.info("Using Adam optimizer with mixed precision")
        else:
            compile_params["optimizer"] = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if loss_type == "binary_crossentropy":
            compile_params["loss"] = binary_cross_entropy_loss
        elif loss_type == "focal_loss":
            gamma = param.get("model", {}).get("focal_loss", {}).get("gamma", 2.0)
            alpha = param.get("model", {}).get("focal_loss", {}).get("alpha", 0.25)
            compile_params["loss"] = lambda y_true, y_pred: focal_loss(y_true, y_pred, gamma, alpha)
        else:
            compile_params["loss"] = "binary_crossentropy"
        
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
                    #Too many labels, need to reduce
                    train_labels_expanded = train_labels_expanded[:train_data.shape[0]]
                    sample_weights = sample_weights[:train_data.shape[0]]
                    logger.info(f"Reduced y to match X: {train_labels_expanded.shape}")

            # Apply mixup augmentation if enabled
            if param.get("training", {}).get("mixup", {}).get("enabled", False):
                alpha = param["training"]["mixup"].get("alpha", 0.2)
                logger.info(f"Applying mixup augmentation with alpha={alpha}")
                train_data, train_labels_expanded = mixup_data(
                    train_data, train_labels_expanded, alpha=alpha
                )

            # Train the model
            history = model.fit(
                train_data,
                train_labels_expanded,
                batch_size=batch_size,
                epochs=epochs,
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
        y_pred_binary = (y_pred > 0.5).astype(int)

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