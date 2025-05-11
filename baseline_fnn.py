#!/usr/bin/env python
"""
 @file   baseline_fnn.py
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
import optuna

from functools import partial
from tqdm import tqdm
from sklearn import metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, Add, MultiHeadAttention
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.regularizers import l2
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from pathlib import Path
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

def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         augment=False,
                         param=None):
    """
    Convert file_name to a vector array with optional augmentation for normal data.
    """
    dims = n_mels * frames
    
    try:
        sr, y = demux_wav(file_name)
        if y is None:
            logger.error(f"Failed to load {file_name}")
            return np.empty((0, dims), float)
        
        # Skip files that are too short
        if len(y) < n_fft:
            logger.warning(f"File too short: {file_name}")
            return np.empty((0, dims), float)
            
        mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                       sr=sr,
                                                       n_fft=n_fft,
                                                       hop_length=hop_length,
                                                       n_mels=n_mels,
                                                       power=power)
        log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
        
        if augment and param is not None and param.get("feature", {}).get("augmentation", {}).get("enabled", False):
            log_mel_spectrogram = augment_spectrogram(log_mel_spectrogram, param)
        
        vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1
        if vectorarray_size < 1:
            logger.warning(f"Not enough frames in {file_name}")
            return np.empty((0, dims), float)

        # Use a more memory-efficient approach
        vectorarray = np.zeros((vectorarray_size, dims), float)
        for t in range(frames):
            vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

        # Normalize for binary cross-entropy
        if np.max(vectorarray) > np.min(vectorarray):
            vectorarray = (vectorarray - np.min(vectorarray)) / (np.max(vectorarray) - np.min(vectorarray) + sys.float_info.epsilon)
            
        # Apply subsampling to reduce memory usage
        stride = param.get("feature", {}).get("stride", 4) if param else 4
        vectorarray = vectorarray[::stride, :]
        
        return vectorarray
        
    except Exception as e:
        logger.error(f"Error in file_to_vector_array for {file_name}: {e}")
        return np.empty((0, dims), float)

def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         augment=False):
    dims = n_mels * frames
    dataset = None
    total_size = 0

    for idx in tqdm(range(len(file_list)), desc=msg, total=len(file_list)):
        try:
            vector_array = file_to_vector_array(file_list[idx],
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power,
                                            augment=augment)
        
            if vector_array.shape[0] == 0:
                continue

            if dataset is None:
                dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), float)
            
            dataset[total_size:total_size + vector_array.shape[0], :] = vector_array
            total_size += vector_array.shape[0]

        except Exception as e:
            logger.error(f"Failed to process file {file_list[idx]}: {e}")
            continue  # Skip this file and continue with the next

    if dataset is None:
        logger.warning("No valid data was found in the file list.")
        return np.empty((0, dims), float)
        
    return dataset[:total_size, :]

def list_to_vector_array_with_labels(file_list, labels,
                                      msg="calc...",
                                      n_mels=64,
                                      frames=5,
                                      n_fft=1024,
                                      hop_length=512,
                                      power=2.0,
                                      augment=False,
                                      batch_size=500):  # Added batch_size parameter
    dims = n_mels * frames
    dataset = None
    expanded_labels = None
    total_size = 0
    
    # Process files in batches to avoid memory issues
    for batch_start in tqdm(range(0, len(file_list), batch_size), desc=f"{msg} (in batches)"):
        batch_end = min(batch_start + batch_size, len(file_list))
        batch_files = file_list[batch_start:batch_end]
        batch_labels = labels[batch_start:batch_end]
        
        batch_vectors = []
        batch_expanded_labels = []
        
        for idx, file_path in enumerate(batch_files):
            try:
                vector_array = file_to_vector_array(file_path,
                                                   n_mels=n_mels,
                                                   frames=frames,
                                                   n_fft=n_fft,
                                                   hop_length=hop_length,
                                                   power=power,
                                                   augment=augment)
                
                if vector_array.shape[0] == 0:
                    continue
                
                # Store vectors and corresponding labels
                batch_vectors.append(vector_array)
                batch_expanded_labels.extend([batch_labels[idx]] * vector_array.shape[0])
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        if not batch_vectors:
            continue
            
        # Combine all vectors from this batch
        batch_dataset = np.vstack(batch_vectors) if batch_vectors else np.empty((0, dims), float)
        batch_expanded_labels = np.array(batch_expanded_labels)
        
        # Initialize or extend the main dataset arrays
        if dataset is None:
            dataset = batch_dataset
            expanded_labels = batch_expanded_labels
        else:
            dataset = np.vstack([dataset, batch_dataset])
            expanded_labels = np.concatenate([expanded_labels, batch_expanded_labels])
        
        total_size += batch_dataset.shape[0]
        logger.info(f"Processed batch {batch_start//batch_size + 1}/{(len(file_list)-1)//batch_size + 1}, total samples: {total_size}")

    if dataset is None:
        logger.warning("No valid data was found in the file list.")
        return np.empty((0, dims), float), np.array([])

    return dataset, expanded_labels



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

########################################################################
# keras model
########################################################################
def keras_model(input_dim, config=None):
    """
    Define an enhanced keras model with attention in the bottleneck and weight decay.
    """
    if config is None:
        config = {}
    
    depth = config.get("depth", 3)
    width = config.get("width", 64)
    bottleneck = config.get("bottleneck", 8)
    dropout_rate = config.get("dropout", 0.2)
    use_batch_norm = config.get("batch_norm", False)
    use_residual = config.get("residual", False)
    activation = config.get("activation", "relu")
    weight_decay = config.get("weight_decay", 1e-4)
    
    inputLayer = Input(shape=(input_dim,))
    
    x = Dense(width, activation=None, kernel_regularizer=l2(weight_decay))(inputLayer)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    
    first_layer_output = x
    
    # Encoder
    for i in range(depth - 1):
        layer_width = max(width // (2 ** (i + 1)), bottleneck)
        layer_input = x
        
        x = Dense(layer_width, activation=None, kernel_regularizer=l2(weight_decay))(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        if use_residual and layer_input.shape[-1] == layer_width:
            x = Add()([x, layer_input])
    

    bottleneck_output = Dense(bottleneck, activation=activation, name="bottleneck")(x)
    x = bottleneck_output
    
    # Decoder
    for i in range(depth - 1):
        layer_width = max(bottleneck * (2 ** (i + 1)), width // (2 ** (depth - i - 2)))
        layer_input = x
        
        x = Dense(layer_width, activation=None, kernel_regularizer=l2(weight_decay))(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        if use_residual and layer_input.shape[-1] == layer_width:
            x = Add()([x, layer_input])
    
    if use_residual and first_layer_output.shape[-1] == input_dim:
        x = Dense(1, activation="sigmoid", kernel_regularizer=l2(weight_decay))(x)
        output = x
    else:
        output = Dense(1, activation="sigmoid", kernel_regularizer=l2(weight_decay))(x)

    return Model(inputs=inputLayer, outputs=output)


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
# Optuna integration
########################################################################
def objective(trial, param, x_train, y_train, x_val, y_val):
    """
    Define the objective function for Optuna optimization.
    """
    # Define the hyperparameter search space
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    depth = trial.suggest_int('depth', 2, 6)
    width = trial.suggest_int('width', 64, 256)

    # Build the model
    inputs = Input(shape=(x_train.shape[1],))
    x = inputs
    for _ in range(depth):
        x = Dense(width, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=param['fit']['epochs'],
        batch_size=param['fit']['batch_size'],
        verbose=0
    )

    # Return the validation loss as the objective value
    val_loss = min(history.history['val_loss'])
    return val_loss

########################################################################
# main
########################################################################
def main():

    with open("baseline_fnn.yaml", "r") as stream:
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

    with open("baseline_fnn.yaml", "r") as stream:
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
        train_labels_expanded = train_labels  # Ensure train_labels_expanded is assigned
        val_data = load_pickle(val_pickle)
        val_labels = load_pickle(val_labels_pickle)
        val_labels_expanded = val_labels  # Ensure val_labels_expanded is assigned
        test_files = load_pickle(test_files_pickle)
        test_labels = load_pickle(test_labels_pickle)
        print('DEBUG: Loaded preprocessed data from pickle files.')

    else:
        train_files, train_labels, val_files, val_labels, test_files, test_labels = dataset_generator(target_dir, param=param)

        if len(train_files) == 0 or len(val_files) == 0 or len(test_files) == 0:
            logger.error(f"No files found for {evaluation_result_key}, skipping...")
            return  # Exit main() if no files are found after generation

        train_data, train_labels_expanded = list_to_vector_array_with_labels(
            train_files,
            train_labels,
            msg="generate train_dataset",
            n_mels=param["feature"]["n_mels"],
            frames=param["feature"]["frames"],
            n_fft=param["feature"]["n_fft"],
            hop_length=param["feature"]["hop_length"],
            power=param["feature"]["power"],
            augment=True
        )
        print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels_expanded.shape}")

        val_data, val_labels_expanded = list_to_vector_array_with_labels(
            val_files,
            val_labels,
            msg="generate validation_dataset",
            n_mels=param["feature"]["n_mels"],
            frames=param["feature"]["frames"],
            n_fft=param["feature"]["n_fft"],
            hop_length=param["feature"]["hop_length"],
            power=param["feature"]["power"],
            augment=False
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

    print("============== OPTUNA OPTIMIZATION ==============")
    if param.get("optuna", {}).get("enabled", False):
        def objective(trial):
            return objective(trial, param=param, x_train=train_data, y_train=train_labels_expanded, x_val=val_data, y_val=val_labels_expanded)

        study = optuna.create_study(direction='minimize')
        study.optimize(partial(objective, param=param, x_train=train_data, y_train=train_labels_expanded, x_val=val_data, y_val=val_labels_expanded),
                       n_trials=param["optuna"].get("trials", 50))

        # Log the best hyperparameters
        logger.info(f"Best hyperparameters: {study.best_params}")

        # Save the best hyperparameters to a YAML file
        best_hyperparameters_file = f"{param['result_directory']}/best_hyperparameters.yaml"
        with open(best_hyperparameters_file, 'w') as f:
            yaml.dump(study.best_params, f)

        print(f"Optuna optimization completed. Best hyperparameters saved to {best_hyperparameters_file}")
    else:
        print("Optuna optimization is disabled in the configuration.")

    print("============== MODEL TRAINING ==============")
    # Track model training time specifically
    model_start_time = time.time()
    # Define model_file and history_img variables
    model_file = f"{param['model_directory']}/model_overall.keras"
    history_img = f"{param['result_directory']}/history_overall.png"

    model_config = param.get("model", {}).get("architecture", {})
    model = keras_model(
        param["feature"]["n_mels"] * param["feature"]["frames"],
        config=model_config
    )
    model.summary()

    # Always train the model, ignoring any pre-existing model file
    compile_params = param["fit"]["compile"].copy()
    loss_type = param.get("model", {}).get("loss", "mse")

    # Handle learning_rate separately for the optimizer
    learning_rate = compile_params.pop("learning_rate", 0.001) if "learning_rate" in compile_params else 0.001

    if loss_type == "binary_crossentropy":
        compile_params["loss"] = binary_cross_entropy_loss
    else:
        # Default to standard binary_crossentropy from Keras
        compile_params["loss"] = "binary_crossentropy"

    if "optimizer" in compile_params and compile_params["optimizer"] == "adam":
        compile_params["optimizer"] = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Add weighted_metrics to fix the warning
    if "weighted_metrics" not in compile_params:
        compile_params["weighted_metrics"] = []

    compile_params["metrics"] = ['accuracy']
    model.compile(**compile_params)

    callbacks = []
    early_stopping_config = param.get("fit", {}).get("early_stopping", {})
    if early_stopping_config.get("enabled", False):
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor=early_stopping_config.get("monitor", "val_loss"),
            patience=early_stopping_config.get("patience", 10),
            min_delta=early_stopping_config.get("min_delta", 0.001),
            restore_best_weights=early_stopping_config.get("restore_best_weights", True)
        ))

    sample_weights = np.ones(len(train_data))

    # Apply uniform sample weights - no machine-specific weighting
    if param.get("fit", {}).get("apply_sample_weights", False):
        # Apply a uniform weight factor if needed
        weight_factor = param.get("fit", {}).get("weight_factor", 1.0)
        sample_weights *= weight_factor

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

    print("============== EVALUATION ==============")
    y_pred = []
    y_true = test_labels

    for num, file_name in tqdm(enumerate(test_files), total=len(test_files)):
        try:
            data = file_to_vector_array(file_name,
                                    n_mels=param["feature"]["n_mels"],
                                    frames=param["feature"]["frames"],
                                    n_fft=param["feature"]["n_fft"],
                                    hop_length=param["feature"]["hop_length"],
                                    power=param["feature"]["power"],
                                    augment=False)  # No augmentation during eval
                                    
            if data.shape[0] == 0:
                logger.warning(f"No valid features extracted from file: {file_name}")
                continue
                    
            # Get the predicted class probability
            pred = model.predict(data, verbose=0)
            # Average the predictions across all frames
            file_pred = np.mean(pred)
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

    # Save the optimal threshold to the evaluation result
    evaluation_result["OptimalThreshold"] = float(optimal_threshold)

    # Save results to YAML file
    with open(result_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)

    # Convert predictions to binary using optimal threshold
    y_pred_binary = (np.array(y_pred) >= optimal_threshold).astype(int)

    # Plot and save confusion matrix
    visualizer.plot_confusion_matrix(y_true, y_pred_binary)
    visualizer.save_figure(f"{param['result_directory']}/confusion_matrix_{evaluation_result_key}.png")

    # Calculate metrics
    accuracy = metrics.accuracy_score(y_true, y_pred_binary)

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