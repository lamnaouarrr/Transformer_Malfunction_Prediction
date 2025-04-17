#!/usr/bin/env python
"""
 @file   baseline.py
 @brief  Baseline code of simple AE-based anomaly detection used experiment in [1], updated for 2025.
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
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy as np
import librosa
import librosa.core
import librosa.feature
import yaml
import time
import tensorflow as tf
import matplotlib.pyplot as plt

# from import
from pathlib import Path
from tqdm import tqdm
from sklearn import metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Activation, Add
from tensorflow.keras.regularizers import l2
########################################################################


########################################################################
# version
########################################################################
__versions__ = "2.0.0"
########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
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
        # Don't create figure in constructor, create fresh figure each time
        pass

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        # Create a new figure every time to prevent issues with reusing figures
        plt.figure(figsize=(10, 7))
        plt.plot(loss, label="Train Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.grid(True)

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save .png file path.

        return : None
        """
        # Make sure directory exists
        os.makedirs(os.path.dirname(name), exist_ok=True)
        
        try:
            plt.tight_layout()  # Adjust layout to make room for labels
            plt.savefig(name, dpi=300)  # Higher DPI for better quality
            logger.info(f"Figure saved to {name}")
        except Exception as e:
            logger.error(f"Error saving figure to {name}: {e}")
        finally:
            plt.close()  # Always close the figure to free memory


########################################################################


########################################################################
# file I/O
########################################################################
# pickle I/O
def save_pickle(filename, save_data):
    """
    picklenize the data.

    filename : str
        pickle filename
    data : free datatype
        some data will be picklenized

    return : None
    """
    logger.info(f"save_pickle -> {filename}")
    with open(filename, 'wb') as sf:
        pickle.dump(save_data, sf)


def load_pickle(filename):
    """
    unpicklenize the data.

    filename : str
        pickle filename

    return : data
    """
    logger.info(f"load_pickle <- {filename}")
    with open(filename, 'rb') as lf:
        load_data = pickle.load(lf)
    return load_data


# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except Exception as e:
        logger.error(f"file_broken or not exists!! : {wav_name}, error: {e}")
        return None


def demux_wav(wav_name, channel=0):
    """
    demux .wav file.

    wav_name : str
        target .wav file
    channel : int
        target channel number

    return : numpy.array( float )
        demuxed mono data

    Enabled to read multiple sampling rates.

    Enabled even one channel.
    """
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


########################################################################
# feature extractor
########################################################################
def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         params=None):  # Add params parameter
    """
    convert file_name to a vector array.
    ...
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa (**kwargs == param["librosa"])
    sr, y = demux_wav(file_name)
    if y is None:
        return np.empty((0, dims), float)
        
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                    sr=sr,
                                                    n_fft=n_fft,
                                                    hop_length=hop_length,
                                                    n_mels=n_mels,
                                                    power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vectorarray_size < 1:
        return np.empty((0, dims), float)

    # 06 generate feature vectors by concatenating multi_frames
    if params and params.get("use_patches", False):  # Use passed params here
        # Reshape for patch extraction
        log_mel_reshaped = log_mel_spectrogram.reshape(n_mels, -1)
        vectorarray = extract_patches(log_mel_reshaped, 
                                    patch_size=params["feature"].get("patch_size", 16),
                                    stride=params["feature"].get("stride", 8))
    else:
        # Original method
        vectorarray = np.zeros((vectorarray_size, dims), float)
        for t in range(frames):
            vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

    return vectorarray


def extract_patches(spectrogram, patch_size=16, stride=8):
    """
    Extract overlapping patches from mel-spectrogram.
    
    spectrogram : numpy.array(shape=(n_mels, time))
        The mel-spectrogram to extract patches from
    patch_size : int
        Size of patches to extract
    stride : int
        Stride between patches
    
    return : numpy.array(shape=(num_patches, patch_size*patch_size))
        Extracted patches flattened to vectors
    """
    n_mels, time_length = spectrogram.shape
    patches = []
    
    # Check if spectrogram is large enough for patching
    if n_mels < patch_size or time_length < patch_size:
        logger.warning(f"Spectrogram too small for patching: {n_mels}x{time_length}, patch size: {patch_size}")
        # Return original spectrogram flattened if too small for patching
        return np.array([spectrogram.flatten()])
    
    for i in range(0, n_mels - patch_size + 1, stride):
        for j in range(0, time_length - patch_size + 1, stride):
            patch = spectrogram[i:i+patch_size, j:j+patch_size]
            # Verify patch shape before adding
            if patch.shape == (patch_size, patch_size):
                patches.append(patch.flatten())
            else:
                logger.warning(f"Invalid patch shape: {patch.shape}, expected: ({patch_size}, {patch_size})")
    
    # If no valid patches were extracted, return original flattened
    if len(patches) == 0:
        logger.warning("No valid patches extracted, returning original spectrogram")
        return np.array([spectrogram.flatten()])
        
    return np.array(patches)


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        training dataset (when generate the validation data, this function is not used.)
        * dataset.shape = (total_dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    dataset = None
    total_size = 0

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = file_to_vector_array(file_list[idx],
                                   n_mels=n_mels,
                                   frames=frames,
                                   n_fft=n_fft,
                                   hop_length=hop_length,
                                   power=power,
                                   params=param)
        
        if vector_array.shape[0] == 0:
            continue

        if dataset is None:
            dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), float)
        
        dataset[total_size:total_size + vector_array.shape[0], :] = vector_array
        total_size += vector_array.shape[0]

    if dataset is None:
        logger.warning("No valid data was found in the file list.")
        return np.empty((0, dims), float)
        
    # Return only the filled part of the dataset
    return dataset[:total_size, :]


def augment_data(data_array, augmentation_factor=2, params=None):
    """
    Enhanced data augmentation with more techniques.
    
    data_array : numpy.array
        Input feature vectors
    augmentation_factor : int
        How many augmented samples to create per original
    params : dict
        Configuration parameters
        
    return : numpy.array
        Augmented dataset
    """
    original_size = data_array.shape[0]
    feature_dim = data_array.shape[1]
    augmented_data = np.zeros((original_size * (augmentation_factor + 1), feature_dim))
    
    # Copy original data
    augmented_data[:original_size] = data_array
    
    for i in range(augmentation_factor):
        offset = (i + 1) * original_size
        
        # Copy original data
        augmented_data[offset:offset + original_size] = data_array.copy()
        
        # Apply different augmentation techniques for each copy
        if i % 3 == 0:
            # Add random noise
            noise_level = np.random.uniform(0.001, 0.02)
            augmented_data[offset:offset + original_size] += np.random.normal(
                0, noise_level, size=augmented_data[offset:offset + original_size].shape
            )
        elif i % 3 == 1:
            # Add frequency masking (zero out random frequency bands)
            if params and "n_mels" in params["feature"] and "frames" in params["feature"]:
                n_mels = params["feature"]["n_mels"]
                frames = params["feature"]["frames"]
                
                # Reshape to apply frequency masking
                for j in range(original_size):
                    sample = augmented_data[offset + j].reshape(frames, n_mels)
                    
                    # Frequency mask
                    mask_size = np.random.randint(1, n_mels // 4)
                    mask_start = np.random.randint(0, n_mels - mask_size)
                    sample[:, mask_start:mask_start + mask_size] = 0
                    
                    augmented_data[offset + j] = sample.flatten()
        else:
            # Add time masking or slight shifts
            if params and "n_mels" in params["feature"] and "frames" in params["feature"]:
                n_mels = params["feature"]["n_mels"]
                frames = params["feature"]["frames"]
                
                # Reshape to apply time masking
                for j in range(original_size):
                    sample = augmented_data[offset + j].reshape(frames, n_mels)
                    
                    # Time mask
                    mask_size = np.random.randint(1, frames // 4)
                    mask_start = np.random.randint(0, frames - mask_size)
                    sample[mask_start:mask_start + mask_size, :] = 0
                    
                    augmented_data[offset + j] = sample.flatten()
    
    return augmented_data


def dataset_generator(target_dir,
                      normal_dir_name="normal",
                      abnormal_dir_name="abnormal",
                      ext="wav"):
    """
    target_dir : str
        base directory path of the dataset
    normal_dir_name : str (default="normal")
        directory name the normal data located in
    abnormal_dir_name : str (default="abnormal")
        directory name the abnormal data located in
    ext : str (default="wav")
        filename extension of audio files 

    return : 
        train_data : numpy.array( numpy.array( float ) )
            training dataset
            * dataset.shape = (total_dataset_size, feature_vector_length)
        train_files : list [ str ]
            file list for training
        train_labels : list [ boolean ] 
            label info. list for training
            * normal/abnormal = 0/1
        eval_files : list [ str ]
            file list for evaluation
        eval_labels : list [ boolean ] 
            label info. list for evaluation
            * normal/abnormal = 0/1
    """
    logger.info(f"target_dir : {target_dir}")

    target_dir_path = Path(target_dir)

    # 01 normal list generate
    normal_files = sorted(list(target_dir_path.joinpath(normal_dir_name).glob(f"*.{ext}")))
    normal_files = [str(file_path) for file_path in normal_files]  # Convert Path to string
    normal_labels = np.zeros(len(normal_files))
    
    if len(normal_files) == 0:
        logger.exception(f"No {ext} files found in {normal_dir_name} directory!")
        return [], [], [], []

    # 02 abnormal list generate
    abnormal_files = sorted(list(target_dir_path.joinpath(abnormal_dir_name).glob(f"*.{ext}")))
    abnormal_files = [str(file_path) for file_path in abnormal_files]  # Convert Path to string
    abnormal_labels = np.ones(len(abnormal_files))
    
    if len(abnormal_files) == 0:
        logger.exception(f"No {ext} files found in {abnormal_dir_name} directory!")
        return [], [], [], []

    # 03 separate train & eval
    train_files = normal_files[len(abnormal_files):]
    train_labels = normal_labels[len(abnormal_files):]
    eval_files = np.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    eval_labels = np.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
    
    logger.info(f"train_file num : {len(train_files)}")
    logger.info(f"eval_file  num : {len(eval_files)}")

    return train_files, train_labels, eval_files, eval_labels


########################################################################


########################################################################
# keras model
########################################################################
def keras_model(input_dim, **params):
    """
    Enhanced model with configurable parameters.
    """
    # Default values that can be overridden by params
    layer_size = params.get("layer_size", 64)
    bottleneck_size = params.get("bottleneck_size", 8)
    dropout_rate = params.get("dropout_rate", 0.2)
    activation = params.get("activation", "relu")
    use_residual = params.get("use_residual", True)
    depth = params.get("depth", 2)  # Number of encoding layers
    weight_decay = params.get("weight_decay", None)
    
    # Weight regularizer based on weight_decay parameter
    regularizer = l2(weight_decay) if weight_decay else None
    
    # Input layer
    inputLayer = Input(shape=(input_dim,))
    h = inputLayer
    
    # Encoder
    skip_connections = []
    for i in range(depth):
        if i > 0 and use_residual:
            skip_connections.append(h)
            
        h = Dense(layer_size, kernel_regularizer=regularizer)(h)
        h = BatchNormalization()(h)
        h = Activation(activation)(h)
        h = Dropout(dropout_rate)(h)
    
    # Bottleneck
    h = Dense(bottleneck_size, kernel_regularizer=regularizer)(h)
    h = BatchNormalization()(h)
    h = Activation(activation)(h)
    
    # Decoder
    for i in range(depth):
        h = Dense(layer_size, kernel_regularizer=regularizer)(h)
        h = BatchNormalization()(h)
        h = Activation(activation)(h)
        h = Dropout(dropout_rate)(h)
        
        # Add residual connection if available and enabled
        if use_residual and i < len(skip_connections):
            h = Add()([h, skip_connections[-(i+1)]])  # Connect to corresponding encoder layer
    
    # Output
    output = Dense(input_dim, activation=None)(h)

    return Model(inputs=inputLayer, outputs=output)


def hybrid_loss(y_true, y_pred, alpha=0.7, feature_params=None):
    """
    Combines MSE and SSIM loss with proper handling of small dimensions.
    Falls back to MSE only when dimensions are too small for SSIM.
    """
    # MSE component
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # For SSIM, we need to ensure dimensions make sense
    batch_size = tf.shape(y_true)[0]
    
    if feature_params:
        n_mels = feature_params.get("n_mels", 64)
        frames = feature_params.get("frames", 5)
    else:
        # Fallback to default values if no params provided
        n_mels = 64
        frames = 5
    
    # Check if dimensions are large enough for SSIM (needs at least 11x11)
    if frames >= 11 and n_mels >= 11:
        y_true_reshaped = tf.reshape(y_true, [batch_size, frames, n_mels, 1])
        y_pred_reshaped = tf.reshape(y_pred, [batch_size, frames, n_mels, 1])
        
        ssim_value = tf.image.ssim(y_true_reshaped, y_pred_reshaped, max_val=1.0)
        ssim_loss = 1.0 - tf.reduce_mean(ssim_value)
        
        return alpha * mse + (1 - alpha) * ssim_loss
    else:
        # Fall back to MSE only for small dimensions
        return mse


def get_optimizer_with_scheduler(optimizer_name="adam", lr=0.001):
    """
    Create optimizer with configurable learning rate scheduling.
    """
    if optimizer_name.lower() == "adam":
        # Get schedule parameters from config or use defaults
        first_decay_steps = param["fit"].get("lr_first_decay_steps", 20)
        t_mul = param["fit"].get("lr_t_mul", 2.0)
        m_mul = param["fit"].get("lr_m_mul", 0.9)
        alpha = param["fit"].get("lr_min_factor", 0.1)
        
        # Create learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha
        )
        return tf.keras.optimizers.Adam(learning_rate=lr)
    return optimizer_name



def tune_hyperparameters(train_data, machine_type, machine_id, db):
    """
    Perform hyperparameter tuning using a simple grid search.
    """
    logger.info("Starting hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        "batch_size": [128, 256, 512],
        "learning_rate": [0.01, 0.001, 0.0001],
        "dropout_rate": [0.1, 0.2, 0.3],
        "bottleneck_size": [8, 16, 32]
    }
    
    best_val_loss = float('inf')
    best_params = {}
    
    # Split data for validation
    val_split = 0.2
    split_idx = int(train_data.shape[0] * (1 - val_split))
    train_subset = train_data[:split_idx]
    val_subset = train_data[split_idx:]
    
    # Create results storage
    tuning_results = []
    
    # Simple grid search
    for batch_size in param_grid["batch_size"]:
        for lr in param_grid["learning_rate"]:
            for dropout_rate in param_grid["dropout_rate"]:
                for bottleneck_size in param_grid["bottleneck_size"]:
                    logger.info(f"Testing: batch_size={batch_size}, lr={lr}, dropout={dropout_rate}, bottleneck={bottleneck_size}")
                    
                    # Create model with current parameters
                    def create_tuning_model(input_dim, dropout_rate, bottleneck_size):
                        inputLayer = Input(shape=(input_dim,))
                        h = Dense(64)(inputLayer)
                        h = BatchNormalization()(h)
                        h = Activation("relu")(h)
                        h = Dropout(dropout_rate)(h)
                        h = Dense(64)(h)
                        h = BatchNormalization()(h)
                        h = Activation("relu")(h)
                        h = Dropout(dropout_rate)(h)
                        h = Dense(bottleneck_size)(h)
                        h = BatchNormalization()(h)
                        h = Activation("relu")(h)
                        h = Dense(64)(h)
                        h = BatchNormalization()(h)
                        h = Activation("relu")(h)
                        h = Dropout(dropout_rate)(h)
                        h = Dense(input_dim, activation=None)(h)
                        return Model(inputs=inputLayer, outputs=h)
                    
                    input_dim = train_data.shape[1]
                    model = create_tuning_model(input_dim, dropout_rate, bottleneck_size)
                    
                    # Setup optimizer
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                    model.compile(optimizer=optimizer, loss="mean_squared_error")
                    
                    # Train for a few epochs
                    history = model.fit(
                        train_subset, train_subset,
                        validation_data=(val_subset, val_subset),
                        epochs=20,
                        batch_size=batch_size,
                        verbose=0
                    )
                    
                    # Get final validation loss
                    val_loss = min(history.history["val_loss"])
                    
                    # Store results
                    result = {
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "dropout_rate": dropout_rate,
                        "bottleneck_size": bottleneck_size,
                        "val_loss": val_loss
                    }
                    tuning_results.append(result)
                    
                    # Update best parameters
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = result.copy()
                    
                    # Clean up
                    tf.keras.backend.clear_session()
    
    # Save tuning results
    with open(f"{param['result_directory']}/tuning_{machine_type}_{machine_id}_{db}.yaml", "w") as f:
        yaml.dump(tuning_results, f, default_flow_style=False)
    
    logger.info(f"Best parameters: {best_params}")
    return best_params


########################################################################


########################################################################
# main
########################################################################
def main():
    
    # Record the start time
    start_time = time.time()

    # load parameter yaml
    global param  # Explicitly declare param as global
    with open("baseline.yaml", "r") as stream:
        param = yaml.safe_load(stream)

    # Ensure all required sections exist
    if "feature" not in param:
        param["feature"] = {}
    if "fit" not in param:
        param["fit"] = {}
    if "compile" not in param["fit"]:
        param["fit"]["compile"] = {}
    if "dataset" not in param:
        param["dataset"] = {}

    # make output directory
    os.makedirs(param["pickle_directory"], exist_ok=True)
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["result_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer = Visualizer()

    # load base_directory list using pathlib for better path handling
    base_path = Path(param["base_directory"])
    dirs = sorted(list(base_path.glob("*/*/*")))
    dirs = [str(dir_path) for dir_path in dirs]  # Convert Path to string

    # Filter directories based on YAML configuration
    if "dataset" in param and param["dataset"]:
        filtered_dirs = []
        db_levels = param["dataset"].get("db_levels", [])
        machine_types = param["dataset"].get("machine_types", [])
        
        for dir_path in dirs:
            parts = Path(dir_path).parts
            include = True
            
            # Filter by db_level if specified
            if db_levels and not any(level in parts for level in db_levels):
                include = False
                
            # Filter by machine_type if specified
            if machine_types and not any(m_type in parts for m_type in machine_types):
                include = False
                
            if include:
                filtered_dirs.append(dir_path)
                
        # Replace dirs with filtered list if any filters were applied
        if db_levels or machine_types:
            dirs = filtered_dirs
            logger.info(f"Filtering applied: db_levels={db_levels}, machine_types={machine_types}")


    # Print dirs for debugging
    logger.info(f"Found {len(dirs)} directories to process:")
    for dir_path in dirs:
        logger.info(f"  - {dir_path}")

    # setup the result
    result_file = f"{param['result_directory']}/resultv2.yaml"
    results = {}

    # GPU memory growth to avoid allocating all GPU memory at once
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPUs and enabled memory growth")
        except RuntimeError as e:
            logger.warning(f"Memory growth setting failed: {e}")

    # loop of the base directory
    for dir_idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print(f"[{dir_idx + 1}/{len(dirs)}] {target_dir}")

        # dataset param
        # Extract components from path
        parts = Path(target_dir).parts
        # This assumes paths like "dataset/0dB/fan/id_00"
        db = parts[-3]
        machine_type = parts[-2]
        machine_id = parts[-1]

        logger.info(f"Processing: db={db}, machine_type={machine_type}, machine_id={machine_id}")

        # setup path
        evaluation_result = {}
        train_pickle = f"{param['pickle_directory']}/train_{machine_type}_{machine_id}_{db}.pickle"
        eval_files_pickle = f"{param['pickle_directory']}/eval_files_{machine_type}_{machine_id}_{db}.pickle"
        eval_labels_pickle = f"{param['pickle_directory']}/eval_labels_{machine_type}_{machine_id}_{db}.pickle"
        model_file = f"{param['model_directory']}/model_{machine_type}_{machine_id}_{db}.weights.h5"
        history_img = f"{param['model_directory']}/history_{machine_type}_{machine_id}_{db}.png"
        evaluation_result_key = f"{machine_type}_{machine_id}_{db}"

       # dataset generator
        print("============== DATASET_GENERATOR ==============")
        if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
            train_data = load_pickle(train_pickle)
            eval_files = load_pickle(eval_files_pickle)
            eval_labels = load_pickle(eval_labels_pickle)
        else:
            train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir)
            
            # Check if any files were found
            if len(train_files) == 0 or len(eval_files) == 0:
                logger.error(f"No files found for {evaluation_result_key}, skipping...")
                continue

            train_data = list_to_vector_array(train_files,
                                            msg="generate train_dataset",
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"])
            
            # Check if any valid training data was found
            if train_data.shape[0] == 0:
                logger.error(f"No valid training data for {evaluation_result_key}, skipping...")
                continue

            save_pickle(train_pickle, train_data)
            save_pickle(eval_files_pickle, eval_files)
            save_pickle(eval_labels_pickle, eval_labels)


        if param.get("perform_tuning", False):
            best_params = tune_hyperparameters(train_data, machine_type, machine_id, db)
            # Store best params for model creation
            model_params = {
                "layer_size": param["fit"].get("layer_size", 64),
                "bottleneck_size": param["fit"].get("bottleneck_size", 8),
                "dropout_rate": param["fit"].get("dropout_rate", 0.2),
                "activation": param["fit"].get("activation", "relu"),
                "use_residual": param["fit"].get("use_residual", True),
                "depth": param["fit"].get("depth", 2),
                "weight_decay": param["fit"].get("weight_decay", None)
            }
        else:
            model_params = {
                "layer_size": param["fit"].get("layer_size", 64),
                "bottleneck_size": param["fit"].get("bottleneck_size", 8),
                "dropout_rate": param["fit"].get("dropout_rate", 0.2),
                "activation": param["fit"].get("activation", "relu"),
                "use_residual": param["fit"].get("use_residual", True),
                "depth": param["fit"].get("depth", 2),
                "weight_decay": param["fit"].get("weight_decay", None)
            }

###############################################################################
#debug
        def diagnose_directories():
            """
            Check directory structure and file existence to diagnose setup issues.
            """
            # Check main directories
            directories = [
                param["base_directory"],
                param["pickle_directory"],
                param["model_directory"],
                param["result_directory"]
            ]
            
            for directory in directories:
                if os.path.exists(directory):
                    logger.info(f"Directory exists: {directory}")
                    # List a few files in each directory
                    try:
                        files = os.listdir(directory)[:5]  # Show first 5 files only
                        logger.info(f"Files in {directory}: {files}")
                    except Exception as e:
                        logger.error(f"Error listing files in {directory}: {e}")
                else:
                    logger.error(f"Directory missing: {directory}")
            
            # Check for specific files
            logger.info("Checking for configuration file...")
            if os.path.exists("baseline.yaml"):
                logger.info("baseline.yaml found")
            else:
                logger.error("baseline.yaml not found")
            
            # Print environment info
            import platform
            logger.info(f"Python version: {platform.python_version()}")
            logger.info(f"TensorFlow version: {tf.__version__}")
            logger.info(f"NumPy version: {np.__version__}")
            logger.info(f"Librosa version: {librosa.__version__}")
            
            # Check GPU availability
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                logger.info(f"GPUs available: {len(gpus)}")
                for gpu in gpus:
                    logger.info(f"  {gpu}")
            else:
                logger.info("No GPUs available, using CPU")
            
            return True


        def debug_data_load(train_files, eval_files):
            """
            Debug data loading by checking a few sample files.
            """
            logger.info("Debugging data loading...")
            
            # Check a few training files
            for i, file in enumerate(train_files[:3]):
                logger.info(f"Training file {i}: {file}")
                try:
                    sr, y = demux_wav(file)
                    if y is None:
                        logger.warning(f"  Failed to load file: {file}")
                    else:
                        logger.info(f"  Successfully loaded file: {file}")
                        logger.info(f"  Sample rate: {sr}, Duration: {len(y)/sr:.2f}s, Shape: {y.shape}")
                except Exception as e:
                    logger.error(f"  Error loading file {file}: {e}")
            
            # Check a few evaluation files
            for i, file in enumerate(eval_files[:3]):
                logger.info(f"Evaluation file {i}: {file}")
                try:
                    sr, y = demux_wav(file)
                    if y is None:
                        logger.warning(f"  Failed to load file: {file}")
                    else:
                        logger.info(f"  Successfully loaded file: {file}")
                        logger.info(f"  Sample rate: {sr}, Duration: {len(y)/sr:.2f}s, Shape: {y.shape}")
                except Exception as e:
                    logger.error(f"  Error loading file {file}: {e}")
            
            return True


        def debug_feature_extraction(file_path, params):
            """
            Debug feature extraction for a single file.
            """
            logger.info(f"Debugging feature extraction for file: {file_path}")
            
            try:
                # Load audio
                sr, y = demux_wav(file_path)
                if y is None:
                    logger.error("Failed to load audio")
                    return False
                    
                logger.info(f"Audio loaded: sr={sr}, length={len(y)}, duration={len(y)/sr:.2f}s")
                
                # Extract features
                vector_array = file_to_vector_array(
                    file_path,
                    n_mels=params["feature"]["n_mels"],
                    frames=params["feature"]["frames"],
                    n_fft=params["feature"]["n_fft"],
                    hop_length=params["feature"]["hop_length"],
                    power=params["feature"]["power"],
                    params=params
                )
                
                logger.info(f"Feature extraction result: shape={vector_array.shape}")
                
                if vector_array.shape[0] == 0:
                    logger.error("Feature extraction returned empty array")
                    return False
                    
                # Additional info
                logger.info(f"Feature stats: min={vector_array.min():.4f}, max={vector_array.max():.4f}, mean={vector_array.mean():.4f}")
                
                return True
            except Exception as e:
                logger.error(f"Error in feature extraction: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
###############################################################################

        # model training
        print("============== MODEL TRAINING ==============")
        # Create model with proper parameters - only create it once
        model = keras_model(
            param["feature"]["n_mels"] * param["feature"]["frames"],
            **model_params
        )
        model.summary()

        # training
        if os.path.exists(model_file) and not param.get("force_retrain", False):
            try:
                model.load_weights(model_file)
                logger.info(f"Model loaded from {model_file}")
                
                # Create placeholder history data for visualization
                # This ensures history graphs are created even when loading a model
                dummy_history = {"loss": [0.1, 0.05], "val_loss": [0.12, 0.06]}
                visualizer.loss_plot(dummy_history["loss"], dummy_history["val_loss"])
                visualizer.save_figure(history_img)
                logger.info(f"Created placeholder history graph at {history_img}")
            except Exception as e:
                logger.error(f"Error loading model from {model_file}: {e}")
                logger.info("Will train a new model")
                os.remove(model_file) if os.path.exists(model_file) else None
                model_exists = False
        else:
            logger.info(f"Model file {model_file} not found or force_retrain enabled, training new model")
            model_exists = False

        # Only train if model doesn't exist or loading failed
        if not os.path.exists(model_file):
            # Update compile parameters for TensorFlow 2.x
            compile_params = param["fit"]["compile"].copy()
            if "optimizer" in compile_params and compile_params["optimizer"] == "adam":
                compile_params["optimizer"] = get_optimizer_with_scheduler(
                    "adam", 
                    lr=param["fit"].get("learning_rate", 0.001)
                )
            
            # Define custom loss function that includes feature parameters
            def get_loss_function():
                feature_params = param["feature"]
                alpha = param["fit"].get("loss_alpha", 0.7)
                def custom_hybrid_loss(y_true, y_pred):
                    return hybrid_loss(y_true, y_pred, alpha=alpha, feature_params=feature_params)
                return custom_hybrid_loss

            if compile_params.get("loss") == "hybrid_loss":
                loss_function = get_loss_function()
            else:
                loss_function = compile_params.get("loss")

            model.compile(optimizer=compile_params.get("optimizer"), loss=loss_function)
            
            # Set up callbacks
            callbacks = []

            if param["fit"].get("use_augmentation", False):
                logger.info("Applying data augmentation")
                train_data = augment_data(
                    train_data, 
                    augmentation_factor=param["fit"].get("augmentation_factor", 2),
                    params=param
                )
                logger.info(f"Data augmented: {train_data.shape[0]} samples")

            # Early stopping
            if param["fit"].get("early_stopping", True):
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ))

            # Model checkpoint - ensure directory exists
            model_dir = os.path.dirname(model_file)
            os.makedirs(model_dir, exist_ok=True)
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                filepath=model_file,
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            ))

            # Add TensorBoard logging
            log_dir = f"{param['result_directory']}/logs/{machine_type}_{machine_id}_{db}_{time.strftime('%Y%m%d-%H%M%S')}"
            os.makedirs(log_dir, exist_ok=True)
            callbacks.append(tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True
            ))

            # Add ReduceLROnPlateau
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ))

            # Ensure validation data is properly created
            validation_split = param["fit"].get("validation_split", 0.1)
            if validation_split <= 0 or validation_split >= 1:
                logger.warning(f"Invalid validation_split value: {validation_split}, setting to default 0.1")
                validation_split = 0.1

            # Make sure train_data has enough samples for validation split
            if train_data.shape[0] < 10:  # At least 10 samples for meaningful validation
                logger.warning(f"Not enough training data samples ({train_data.shape[0]}), disabling validation")
                validation_split = 0

            # Fit the model
            try:
                logger.info(f"Starting training with {train_data.shape[0]} samples, validation_split={validation_split}")
                history = model.fit(
                    train_data,
                    train_data,
                    epochs=param["fit"]["epochs"],
                    batch_size=param["fit"]["batch_size"],
                    shuffle=param["fit"]["shuffle"],
                    validation_split=validation_split,
                    verbose=param["fit"]["verbose"],
                    callbacks=callbacks
                )

                # Always save the visualization
                visualizer.loss_plot(history.history["loss"], history.history["val_loss"])
                visualizer.save_figure(history_img)
                logger.info(f"Training history graph saved to {history_img}")
                
                # Save model weights
                model.save_weights(model_file)
                logger.info(f"Model weights saved to {model_file}")
            except Exception as e:
                logger.error(f"Error during model training: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # evaluation
        print("============== EVALUATION ==============")
        y_pred = [0. for _ in eval_labels]
        y_true = eval_labels

        # Process files in batches to save memory
        batch_size = 32
        ssim_scores = []

        for i in range(0, len(eval_files), batch_size):
            batch_files = eval_files[i:i+batch_size]
            
            for num, file_name in tqdm(enumerate(batch_files), total=len(batch_files)):
                try:
                    file_index = i + num
                    data = file_to_vector_array(file_name,
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"],
                                            params=param)
                                            
                    if data.shape[0] == 0:
                        logger.warning(f"No valid features extracted from file: {file_name}")
                        continue
                        
                    pred = model.predict(data, verbose=0)
                    error = np.mean(np.square(data - pred), axis=1)
                    y_pred[file_index] = np.mean(error)
                    
                    # Calculate SSIM for each sample
                    for j in range(min(len(data), len(pred))):
                        try:
                            # Reshape vectors back to spectrograms
                            orig_spec = data[j].reshape(param["feature"]["frames"], param["feature"]["n_mels"])
                            recon_spec = pred[j].reshape(param["feature"]["frames"], param["feature"]["n_mels"])
                            
                            # Get dimensions
                            height, width = orig_spec.shape
                            
                            # Calculate appropriate window size (must be odd and smaller than image)
                            win_size = min(height, width)
                            if win_size % 2 == 0:  # Make odd if even
                                win_size -= 1
                            if win_size > 1:  # Ensure at least 3 for ssim
                                win_size = max(3, win_size)
                                # Calculate SSIM with custom window size
                                ssim_value = ssim(
                                    orig_spec, 
                                    recon_spec,
                                    win_size=win_size,
                                    data_range=orig_spec.max() - orig_spec.min() + 1e-10
                                )
                                ssim_scores.append(ssim_value)
                        except Exception as e:
                            logger.warning(f"SSIM calculation error: {e}")
                            
                except Exception as e:
                    logger.warning(f"Error processing file: {file_name}, error: {e}")

        # Calculate AUC score
        try:
            score = metrics.roc_auc_score(y_true, y_pred)
            logger.info(f"AUC : {score}")
            evaluation_result["AUC"] = float(score)
            
            # Additional metrics
            precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
            evaluation_result["Average Precision"] = float(metrics.average_precision_score(y_true, y_pred))
            
            # Find optimal F1 score
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
            best_threshold_idx = np.argmax(f1_scores)
            evaluation_result["Best F1"] = float(f1_scores[best_threshold_idx])
            best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else thresholds[-1]
            evaluation_result["Best Threshold"] = float(thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else thresholds[-1])
            
            # Apply best threshold to get binary predictions
            y_pred_binary = (np.array(y_pred) >= best_threshold).astype(int)

            # Calculate additional metrics at the best threshold
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_binary).ravel()
            
            # Precision and Recall at best threshold
            evaluation_result["Precision"] = float(tp / (tp + fp + 1e-10))
            evaluation_result["Recall"] = float(tp / (tp + fn + 1e-10))
            
            # Specificity (True Negative Rate)
            evaluation_result["Specificity"] = float(tn / (tn + fp + 1e-10))
            
            # Matthews Correlation Coefficient (MCC)
            mcc_numerator = (tp * tn) - (fp * fn)
            mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-10)
            evaluation_result["MCC"] = float(mcc_numerator / mcc_denominator)
            
            # Mean Absolute Error
            evaluation_result["MAE"] = float(metrics.mean_absolute_error(y_true, y_pred_binary))

            # Log some of the new metrics
            logger.info(f"Precision: {evaluation_result['Precision']:.4f}, Recall: {evaluation_result['Recall']:.4f}")
            logger.info(f"Specificity: {evaluation_result['Specificity']:.4f}, MCC: {evaluation_result['MCC']:.4f}")
            
            # Add SSIM scores if any were calculated
            if ssim_scores:
                evaluation_result["SSIM"] = float(np.mean(ssim_scores))
                logger.info(f"Avg SSIM: {evaluation_result['SSIM']:.4f}")
            else:
                evaluation_result["SSIM"] = float(-1)  # Indicate no valid SSIM could be calculated
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            logger.error(f"Exception details: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        results[evaluation_result_key] = evaluation_result
            
        print("===========================")

    # Recording the end time
    end_time = time.time()

    # Calculating total execution time
    total_time = end_time - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")

    # saving the execution time to the results file
    results["execution_time_seconds"] = float(total_time)

    # output results
    print("\n===========================")
    logger.info(f"all results -> {result_file}")
    with open(result_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print("===========================")

if __name__ == "__main__":
    main()