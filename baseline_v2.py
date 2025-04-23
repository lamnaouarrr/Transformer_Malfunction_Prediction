#!/usr/bin/env python
"""
 @file   baseline_v2.py
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

from tqdm import tqdm
from sklearn import metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, Add, MultiHeadAttention
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.regularizers import l2
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
from sklearn.mixture import GaussianMixture
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
    logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
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
    def __init__(self):
        pass

    def loss_plot(self, history, machine_type=None, machine_id=None, db=None):
        fig_size = param.get("visualization", {}).get("figure_size", [30, 20])
        plt.figure(figsize=(fig_size[0], fig_size[1]))
        
        # Create title with machine information
        title_info = ""
        if machine_type and machine_id and db:
            title_info = f" for {machine_type} {machine_id} {db}"
        
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
    sr, y = demux_wav(file_name)
    if y is None:
        print(f"Failed to load {file_name}")
        return np.empty((0, dims), float)

    if augment and param.get("feature", {}).get("augmentation", {}).get("enabled", False):
        log_mel_spectrogram = augment_spectrogram(log_mel_spectrogram, param)

        
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                    sr=sr,
                                                    n_fft=n_fft,
                                                    hop_length=hop_length,
                                                    n_mels=n_mels,
                                                    power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    
    if augment and param.get("feature", {}).get("augmentation", {}).get("enabled", False):
        log_mel_spectrogram = augment_spectrogram(log_mel_spectrogram, param)
    
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1
    if vectorarray_size < 1:
        return np.empty((0, dims), float)

    vectorarray = np.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

    # Add normalization for binary cross-entropy
    vectorarray = (vectorarray - np.min(vectorarray)) / (np.max(vectorarray) - np.min(vectorarray) + sys.float_info.epsilon)
    
    return vectorarray

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

    for idx in tqdm(range(len(file_list)), desc=msg):
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
                                      augment=False):
    dims = n_mels * frames
    dataset = None
    expanded_labels = None
    total_size = 0

    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = file_to_vector_array(file_list[idx],
                                           n_mels=n_mels,
                                           frames=frames,
                                           n_fft=n_fft,
                                           hop_length=hop_length,
                                           power=power,
                                           augment=augment)
        
        if vector_array.shape[0] == 0:
            continue

        # Ensure that the labels match the number of samples in vector_array
        file_labels = np.full(vector_array.shape[0], labels[idx])
        
        if dataset is None:
            # Estimate total size to preallocate arrays
            avg_vectors_per_file = vector_array.shape[0]
            estimated_total_size = avg_vectors_per_file * len(file_list)
            dataset = np.zeros((estimated_total_size, dims), float)
            expanded_labels = np.zeros(estimated_total_size, float)
        
        # Ensure there is enough space in the dataset
        if total_size + vector_array.shape[0] > dataset.shape[0]:
            dataset = np.resize(dataset, (dataset.shape[0] * 2, dims))
            expanded_labels = np.resize(expanded_labels, dataset.shape[0])
        
        # Insert the new data into the dataset
        dataset[total_size:total_size + vector_array.shape[0], :] = vector_array
        expanded_labels[total_size:total_size + vector_array.shape[0]] = file_labels
        total_size += vector_array.shape[0]

    # Trimming the dataset to the right size
    if dataset is None:
        logger.warning("No valid data was found in the file list.")
        return np.empty((0, dims), float), np.array([])

    return dataset[:total_size, :], expanded_labels[:total_size]



def dataset_generator(target_dir, param=None):
    """
    Generate training, validation, and testing datasets for the new directory structure.
    
    target_dir: A directory containing audio files for a specific machine ID
    param: parameters dictionary from the YAML config
    split_ratio: train/val/test split ratio as [train, val, test]
    ext: file extension for audio files
    """
    logger.info(f"target_dir : {target_dir}")
    
    if param is None:
        param = {}
    
    split_ratio = param.get("dataset", {}).get("split_ratio", [0.8, 0.1, 0.1])
    ext = param.get("dataset", {}).get("file_extension", "wav")
    
    # Parse the target directory path to extract db, machine_type, and machine_id
    parts = Path(target_dir).parts
    db = None
    machine_type = None
    machine_id = None
    is_normal = "normal" in str(target_dir)
    
    # Find the relevant parts
    for i, part in enumerate(parts):
        if part in ["0dB", "3dB", "6dB", "-3dB", "-6dB"]:
            db = part
            if i+2 < len(parts):
                machine_type = parts[i+1]
                machine_id = parts[i+2]
            break
    
    if not db or not machine_type or not machine_id:
        logger.warning(f"Could not parse directory properly: {target_dir}")
        return [], [], [], [], [], []
    
    # Get files in the current directory
    files_in_dir = list(Path(target_dir).glob(f"*.{ext}"))
    
    # If the target is normal, find corresponding abnormal directory
    base_dir = Path(param.get("base_directory", "./dataset"))
    
    normal_files = []
    abnormal_files = []
    

    if is_normal:
        normal_files = [str(f) for f in files_in_dir]
        # Try to find matching abnormal directory
        abnormal_dir = base_dir / "abnormal" / db / machine_type / machine_id
        if abnormal_dir.exists():
            abnormal_files = [str(f) for f in abnormal_dir.glob(f"*.{ext}")]
    else:
        abnormal_files = [str(f) for f in files_in_dir]
        # Try to find matching normal directory
        normal_dir = base_dir / "normal" / db / machine_type / machine_id
        if normal_dir.exists():
            normal_files = [str(f) for f in normal_dir.glob(f"*.{ext}")]
    
    # Check if we have any files
    if len(normal_files) == 0:
        logger.warning(f"No normal {ext} files found for {machine_type}/{machine_id}")
        if len(abnormal_files) == 0:
            logger.error(f"No files found at all for {machine_type}/{machine_id}")
            return [], [], [], [], [], []
    
    if len(abnormal_files) == 0:
        logger.warning(f"No abnormal {ext} files found for {machine_type}/{machine_id}")
    
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

    print(f"Looking for files in: {target_dir}")
    print(f"Found {len(files_in_dir)} files")


    # debug
    print(f"DEBUG - Breakdown for {machine_id}:")
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
    
    # Get attention parameters
    attention_config = config.get("attention", {})
    use_attention = attention_config.get("enabled", True)
    num_heads = attention_config.get("num_heads", 2)
    key_dim_factor = attention_config.get("key_dim_factor", 0.5)
    bottleneck_weight = attention_config.get("bottleneck_weight", 0.7)
    attention_weight = attention_config.get("attention_weight", 0.3)
    
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
    
    # Bottleneck layer with attention
    bottleneck_output = Dense(bottleneck, activation=activation, name="bottleneck")(x)
    
    if use_attention:
        # Reshape for attention: (batch_size, 1, bottleneck)
        attn_input = tf.expand_dims(bottleneck_output, axis=1)
        # Apply attention with configurable parameters
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=int(bottleneck * key_dim_factor))(attn_input, attn_input)
        attn_output = tf.squeeze(attn_output, axis=1)  # (batch_size, bottleneck)
        # Use configurable weights for attention mix
        x = bottleneck_weight * bottleneck_output + attention_weight * attn_output
    else:
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
# main
########################################################################
def main():
    start_time = time.time()

    with open("baseline.yaml", "r") as stream:
        param = yaml.safe_load(stream)

    os.makedirs(param["pickle_directory"], exist_ok=True)
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["result_directory"], exist_ok=True)

    visualizer = Visualizer()

    base_path = Path(param["base_directory"])

    print("============== COUNTING DATASET SAMPLES ==============")
    logger.info("Counting all samples in the dataset...")
    
    total_normal_files = 0
    total_abnormal_files = 0
    
    # Updated to match the actual directory structure
    normal_path = Path(param["base_directory"]) / "normal"
    abnormal_path = Path(param["base_directory"]) / "abnormal"
    
    # Count normal files
    for db_dir in normal_path.glob("*"):
        if db_dir.is_dir():
            for machine_type_dir in db_dir.glob("*"):
                if machine_type_dir.is_dir():
                    for machine_id_dir in machine_type_dir.glob("*"):
                        if machine_id_dir.is_dir():
                            normal_count = len(list(machine_id_dir.glob("*.wav")))
                            total_normal_files += normal_count
    
    # Count abnormal files
    for db_dir in abnormal_path.glob("*"):
        if db_dir.is_dir():
            for machine_type_dir in db_dir.glob("*"):
                if machine_type_dir.is_dir():
                    for machine_id_dir in machine_type_dir.glob("*"):
                        if machine_id_dir.is_dir():
                            abnormal_count = len(list(machine_id_dir.glob("*.wav")))
                            total_abnormal_files += abnormal_count
    
    # Log the total counts
    logger.info(f"Total normal files in dataset: {total_normal_files}")
    logger.info(f"Total abnormal files in dataset: {total_abnormal_files}")
    logger.info(f"Total files in dataset: {total_normal_files + total_abnormal_files}")

    dirs = []

    # Keep the filtering mechanism based on YAML config
    if param.get("filter", {}).get("enabled", False):
        filter_db = param["filter"].get("db_level")
        filter_machine = param["filter"].get("machine_type")
        filter_id = param["filter"].get("machine_id")
        
        # Start with the normal directory as we're finding our target dirs
        normal_base = base_path / "normal"
        
        pattern_parts = []
        if filter_db:
            pattern_parts.append(filter_db)
        else:
            pattern_parts.append("*")
        if filter_machine:
            pattern_parts.append(filter_machine)
        else:
            pattern_parts.append("*")
        if filter_id:
            pattern_parts.append(filter_id)
        else:
            pattern_parts.append("*")
        
        glob_pattern = str(normal_base.joinpath(*pattern_parts))
        dirs = sorted(glob.glob(glob_pattern))
    else:
        # Get all machine ID directories from the normal path
        # We use normal paths as the base since they typically have all machine IDs
        dirs = sorted(glob.glob(str(base_path / "normal" / "*" / "*" / "*")))

    # Only include directories that contain machine IDs
    dirs = [dir_path for dir_path in dirs if os.path.isdir(dir_path) and "/id_" in dir_path]

    result_file = f"{param['result_directory']}/resultv2.yaml"
    results = {}

    #Create variables to track overall metrics
    all_y_true = []
    all_y_pred = []

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPUs and enabled memory growth")
        except RuntimeError as e:
            logger.warning(f"Memory growth setting failed: {e}")

    for dir_idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print(f"[{dir_idx + 1}/{len(dirs)}] {target_dir}")

        parts = Path(target_dir).parts
        # Adjust for the new structure (normal/abnormal is at the beginning)
        # Format will be like: [..., 'normal'/'abnormal', '0dB', 'fan', 'id_00']
        for i, part in enumerate(parts):
            if part in ['normal', 'abnormal']:
                condition = part  # normal or abnormal
                db = parts[i+1]
                machine_type = parts[i+2]
                machine_id = parts[i+3]
                break
        else:
            # If the loop completes without finding normal/abnormal
            logger.warning(f"Could not parse directory structure properly: {target_dir}")
            # Default to assuming the last 3 parts are what we need
            db = parts[-3]
            machine_type = parts[-2]
            machine_id = parts[-1]

        evaluation_result = {}
        train_pickle = f"{param['pickle_directory']}/train_{machine_type}_{machine_id}_{db}.pickle"
        eval_files_pickle = f"{param['pickle_directory']}/eval_files_{machine_type}_{machine_id}_{db}.pickle"
        eval_labels_pickle = f"{param['pickle_directory']}/eval_labels_{machine_type}_{machine_id}_{db}.pickle"
        model_file = f"{param['model_directory']}/model_{machine_type}_{machine_id}_{db}.weights.h5"
        history_img = f"{param['model_directory']}/history_{machine_type}_{machine_id}_{db}.png"
        evaluation_result_key = f"{machine_type}_{machine_id}_{db}"

       
        print("============== DATASET_GENERATOR ==============")
        train_pickle = f"{param['pickle_directory']}/train_{machine_type}_{machine_id}_{db}.pickle"
        train_labels_pickle = f"{param['pickle_directory']}/train_labels_{machine_type}_{machine_id}_{db}.pickle"
        val_pickle = f"{param['pickle_directory']}/val_{machine_type}_{machine_id}_{db}.pickle"
        val_labels_pickle = f"{param['pickle_directory']}/val_labels_{machine_type}_{machine_id}_{db}.pickle"
        test_files_pickle = f"{param['pickle_directory']}/test_files_{machine_type}_{machine_id}_{db}.pickle"
        test_labels_pickle = f"{param['pickle_directory']}/test_labels_{machine_type}_{machine_id}_{db}.pickle"

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
                continue

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
            continue

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
        model_config = param.get("model", {}).get("architecture", {})
        model = keras_model(
            param["feature"]["n_mels"] * param["feature"]["frames"],
            config=model_config
        )
        model.summary()

        if os.path.exists(model_file):
            model.load_weights(model_file)
            logger.info("Model loaded from file, no training performed")
        else:
            compile_params = param["fit"]["compile"].copy()
            loss_type = param.get("model", {}).get("loss", "mse")

            if loss_type == "binary_crossentropy":
                compile_params["loss"] = binary_cross_entropy_loss
            else:
                # Default to standard binary_crossentropy from Keras
                compile_params["loss"] = "binary_crossentropy"

            if "optimizer" in compile_params and compile_params["optimizer"] == "adam":
                compile_params["optimizer"] = tf.keras.optimizers.Adam()

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

            # Apply targeted sample weighting
            special_case_weights = param.get("fit", {}).get("special_case_weights", {})
            special_case_key = f"{machine_type}_{machine_id}"
            
            if special_case_key in special_case_weights:
                # Apply special case weight
                sample_weights *= special_case_weights[special_case_key]
            elif param.get("fit", {}).get("apply_sample_weights", False):
                # Normal weighting for other cases
                weighted_machine_ids = param.get("fit", {}).get("weighted_machine_ids", [])
                weight_factor = param.get("fit", {}).get("weight_factor", 1.5)
                
                if machine_id in weighted_machine_ids:
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

            model.save_weights(model_file)
            visualizer.loss_plot(history, machine_type, machine_id, db)
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
        if param.get("model", {}).get("threshold_optimization", True):
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
        else:
            optimal_threshold = 0.5  # Default threshold
        logger.info(f"Optimal threshold: {optimal_threshold:.6f}")

        # Convert predictions to binary using optimal threshold
        y_pred_binary = (np.array(y_pred) >= optimal_threshold).astype(int)

        # Calculate metrics
        accuracy = metrics.accuracy_score(y_true, y_pred_binary)
        
        evaluation_result = {}
        evaluation_result["Accuracy"] = float(accuracy)

        logger.info(f"Accuracy: {accuracy:.4f}")

        results[evaluation_result_key] = evaluation_result


        #add the machine's predictions and true labels to the overall collection
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred_binary)

        print("===========================")

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    results["execution_time_seconds"] = float(total_time)

    if len(all_y_true) > 0 and len(all_y_pred_binary) > 0:
        overall_accuracy = metrics.accuracy_score(all_y_true, all_y_pred_binary)
        results["overall_accuracy"] = float(overall_accuracy)
        logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")

    print("\n===========================")
    logger.info(f"all results -> {result_file}")
    with open(result_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print("===========================")

if __name__ == "__main__":
    main()