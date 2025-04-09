#!/usr/bin/env python
"""
 @file   baseline_transformer.py
 @brief  Transformer-based anomaly detection for MIMII dataset, updated for 2025.
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
from pathlib import Path
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
import gc
# from import
from tqdm import tqdm
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Flatten, Reshape
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
########################################################################


########################################################################
# version
########################################################################
__versions__ = "3.0.0"
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


def ssim_loss(y_true, y_pred):
    # Ensure data range is properly calculated
    data_range = tf.reduce_max(y_true) - tf.reduce_min(y_true)
    data_range = tf.maximum(data_range, 1e-6)  # Ensure non-zero range
    
    # Calculate SSIM with fixed filter size of 3
    ssim_val = tf.image.ssim(
        tf.expand_dims(y_true, -1),  # Add channel dimension
        tf.expand_dims(y_pred, -1),
        max_val=data_range,
        filter_size=3
    )
    
    # Return 1 - SSIM as loss (since we want to minimize loss)
    return 1.0 - tf.reduce_mean(ssim_val)


########################################################################
# visualizer
########################################################################
class Visualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(30, 10))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Test"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save .png file path.

        return : None
        """
        plt.savefig(name)
        plt.close()


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
                         power=2.0):
    """
    Modified to return 3D tensor for transformer input
    """
    # 01 calculate the number of dimensions
    sr, y = demux_wav(file_name)
    if y is None:
        return np.empty((0, frames, n_mels), float)
        
    # Normalize audio signal
    y = librosa.util.normalize(y)
    
    # Generate melspectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                    sr=sr,
                                                    n_fft=n_fft,
                                                    hop_length=hop_length,
                                                    n_mels=n_mels,
                                                    power=power)

    #normalization
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    #Z-score normalization (add epsilon for stability)
    log_mel_spectrogram = (log_mel_spectrogram - log_mel_spectrogram.mean()) / (log_mel_spectrogram.std() + 1e-10)
    # Generate sliding window sequences
    vectorarray_size = log_mel_spectrogram.shape[1] - frames + 1
    
    # Skip too short clips
    if vectorarray_size < 1:
        return np.empty((0, frames, n_mels), float)

    # Create 3D tensor with sliding window
    vectorarray = np.zeros((vectorarray_size, frames, n_mels), float)
    for t in range(vectorarray_size):
        # Reshape to (frames, n_mels) instead of flattening
        vectorarray[t] = log_mel_spectrogram[:, t:t+frames].T

    return vectorarray


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=32,
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
    # Initialize with empty array
    vector_array = np.empty((0, frames, n_mels), float)
    
    for file_name in tqdm(file_list, desc=msg):
        vectors = file_to_vector_array(file_name,
                                      n_mels=n_mels,
                                      frames=frames,
                                      n_fft=n_fft,
                                      hop_length=hop_length,
                                      power=power)
        
        # Skip if no vectors were extracted
        if vectors.shape[0] == 0:
            continue
            
        # Ensure vectors have the correct 3D shape
        if len(vectors.shape) != 3:
            logger.warning(f"Unexpected shape from file_to_vector_array: {vectors.shape}")
            continue
            
        # Concatenate along the first dimension (samples)
        vector_array = np.concatenate((vector_array, vectors), axis=0)
        
    return vector_array


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
    train_labels = normal_labels[len(abnormal_files):]
    train_files = normal_files[len(abnormal_files):]
    eval_normal_files = normal_files[:len(abnormal_files)]
    eval_files = eval_normal_files + abnormal_files
    eval_labels = np.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
    
    logger.info(f"train_file num : {len(train_files)}")
    logger.info(f"eval_file  num : {len(eval_files)}")

    return train_files, train_labels, eval_files, eval_labels


########################################################################


########################################################################
# transformer model
########################################################################
def positional_encoding(seq_length, d_model):
    """
    Generate positional encoding for transformer input
    
    Args:
        seq_length (int): Length of the sequence
        d_model (int): Dimension of the model
    
    Returns:
        numpy.ndarray: Positional encoding matrix
    """
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((seq_length, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return pos_encoding.astype(np.float32)

def transformer_model(input_shape, head_size=64, num_heads=4, ff_dim=128, 
                      num_transformer_blocks=2, mlp_units=[128, 64], dropout=0.2):
    """
    Enhanced Sequence-to-Sequence Transformer Model for Anomaly Detection
    
    Args:
        input_shape (tuple): Shape of input (time_steps, n_mels)
        head_size (int): Size of attention heads
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward dimension
        num_transformer_blocks (int): Number of transformer blocks
        mlp_units (list): Units for final MLP layers
        dropout (float): Dropout rate
    
    Returns:
        Keras Model with sequence-to-sequence reconstruction capability
    """
    # Input layer
    inputs = Input(shape=input_shape)  # Shape: (frames, n_mels)
    x = inputs
    
    # Positional Encoding
    seq_length, d_model = input_shape
    pos_encoding = positional_encoding(seq_length, d_model)
    pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
    x = x + pos_encoding  # Inject positional information & Scale positional encoding
    
    # Encoder Transformer Blocks
    for _ in range(num_transformer_blocks):
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            key_dim=head_size, 
            num_heads=num_heads, 
            dropout=dropout
        )(x, x)
        
        # Residual connection and layer normalization
        x = LayerNormalization()(x + attention_output)
        
        # Feed-forward network
        ff_output = Dense(d_model, activation="relu")(x)
        ff_output = Dropout(dropout)(ff_output)
        
        # Residual connection and layer normalization
        x = LayerNormalization()(x + ff_output)
    
    # Bottleneck representation
    encoded = x
    
    # Decoder Transformer Blocks (for reconstruction)
    x = encoded
    for _ in range(num_transformer_blocks):
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            key_dim=head_size, 
            num_heads=num_heads, 
            dropout=dropout
        )(x, x)
        
        # Residual connection and layer normalization
        x = LayerNormalization()(x + attention_output)
        
        # Feed-forward network
        ff_output = Dense(d_model, activation="relu")(x)
        ff_output = Dropout(dropout)(ff_output)
        
        # Residual connection and layer normalization
        x = LayerNormalization()(x + ff_output)
    
    # Final reconstruction layers
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(dropout)(x)
    
    # Output layer (reconstruct the original sequence)
    outputs = Dense(d_model)(x)
    
    return Model(inputs=inputs, outputs=outputs)



########################################################################

def log_memory_usage(message=""):
    """Log current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logger.info(f"Memory usage {message}: {mem_info.rss / (1024 * 1024):.2f} MB")
    except ImportError:
        logger.warning("psutil not installed, memory usage logging disabled")


########################################################################
# The replacement code compiles the model using parameters from YAML configuration.
# It trains the model with the train_data (normal samples) and saves the weights/callbacks.
# If the model weights fail to load, the code falls back to training a new model.
########################################################################

def compile_and_train_model_efficiently(model, train_data, param, visualizer, history_img, model_file):
    """Memory-efficient model training with data batching and generator-based approach"""
    # Default compilation parameters
    learning_rate = param["fit"].get("learning_rate", 0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    train_data_sampled = train_data
    logger.info(f"Using full training dataset of {len(train_data)} examples")
    
    # Enable mixed precision if available
    try:
        # RTX 4000 Ada has Tensor Cores that can accelerate mixed precision
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled for RTX 4000 Ada")
        
        # Additional optimization for RTX GPUs
        tf.config.optimizer.set_jit(True)  # Enable XLA compilation
        logger.info("XLA compilation enabled")
    except Exception as e:
        logger.warning(f"GPU optimizations not fully available: {e}")
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=ssim_loss,
        metrics=['mae', 'mse']
    )
    
    # Callbacks
    callbacks = []
    if param["fit"].get("early_stopping", False):
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ))
    
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    )
    
    model_file_parts = os.path.basename(model_file).split('_')
    machine_type = model_file_parts[1]
    machine_id = model_file_parts[2]
    db = model_file_parts[3].split('.')[0]

    checkpoint_path = f"{param['model_directory']}/checkpoint_{machine_type}_{machine_id}_{db}"
    os.makedirs(checkpoint_path, exist_ok=True)

    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_path, "model_{epoch:02d}.weights.h5"),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
    )

    batch_size = param["fit"]["batch_size"]
    validation_split = param["fit"]["validation_split"]
    
    # Calculate split point
    split_idx = int(len(train_data_sampled) * (1 - validation_split))
    
    # Create training and validation generators
    def data_generator(data, batch_size, is_training=True):
        indices = np.arange(len(data))
        if is_training:
            np.random.shuffle(indices)
        
        while True:  # Make this an infinite generator for Keras
            for start_idx in range(0, len(indices), batch_size):
                end_idx = min(start_idx + batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                batch_x = data[batch_indices]
                # Apply data augmentation for training
                if is_training and np.random.random() < 0.5:
                    # Add random noise + time masking
                    noise = np.random.normal(0, 0.01, batch_x.shape)
                    batch_x_aug = batch_x + noise
                    
                    # Optional: frequency masking (randomly mask some frequency bands)
                    if np.random.random() < 0.3:
                        freq_mask_size = np.random.randint(1, 10)
                        freq_start = np.random.randint(0, batch_x.shape[2] - freq_mask_size)
                        batch_x_aug[:, :, freq_start:freq_start+freq_mask_size] = 0
                    
                    yield batch_x_aug, batch_x
                else:
                    yield batch_x, batch_x
    
    # Calculate steps per epoch
    steps_per_epoch = split_idx // batch_size
    validation_steps = (len(train_data_sampled) - split_idx) // batch_size
    
    # Free up memory
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Train with generator
    history = model.fit(
        data_generator(train_data_sampled[:split_idx], batch_size),
        epochs=param["fit"]["epochs"],
        steps_per_epoch=steps_per_epoch,
        validation_data=data_generator(train_data_sampled[split_idx:], batch_size, is_training=False),
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save artifacts
    visualizer.loss_plot(history.history["loss"], history.history.get("val_loss", []))
    visualizer.save_figure(history_img)
    model.save_weights(model_file)
    
    return history

########################################################################
# main
########################################################################
def main():

    # Record the start time
    start_time = time.time()

    # load parameter yaml
    with open("baseline_transformer.yaml", "r") as stream:
        param = yaml.safe_load(stream)

    #THE EXPERIMENT TRACKING CODE
    experiment_name = f"transformer_ssim_{int(time.time())}"
    results = {}  # Initialize results dictionary
    results["experiment_info"] = {
        "name": experiment_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_config": param["feature"],
        "transformer_config": param["transformer"],
        "fit_config": param["fit"],
        "notes": "RTX 4000 Ada run with memory optimizations"
    }

    # make output directory
    os.makedirs(param["pickle_directory"], exist_ok=True)
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["result_directory"], exist_ok=True)

    # Set TensorFlow memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            # For RTX GPUs, set memory limit to leave some memory for system
            if len(gpus) > 0:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=14000)]  # 14GB limit
                )
            logger.info(f"Found {len(gpus)} GPUs with memory growth enabled")
        except RuntimeError as e:
            logger.warning(f"Memory growth setting failed: {e}")

    # initialize the visualizer
    visualizer = Visualizer()

    # load base_directory list using pathlib for better path handling
    base_path = Path(param["base_directory"])
    dirs = sorted(list(base_path.glob("*/*/*")))
    dirs = [str(dir_path) for dir_path in dirs]  # Convert Path to string

    # Print dirs for debugging
    logger.info(f"Found {len(dirs)} directories to process:")
    for dir_path in dirs:
        logger.info(f"  - {dir_path}")

    # setup the result
    result_file = f"{param['result_directory']}/result_transformer.yaml"
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
        #debug
        log_memory_usage("before data loading")

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
        #debug
        log_memory_usage("before loading data for evaluation")
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
            

            #debuging
            print("Train data shape:", train_data.shape)
            print("First sample shape:", train_data[0].shape if len(train_data) > 0 else "No data")

            # Check if any valid training data was found
            if train_data.shape[0] == 0:
                logger.error(f"No valid training data for {evaluation_result_key}, skipping...")
                continue

            save_pickle(train_pickle, train_data)
            save_pickle(eval_files_pickle, eval_files)
            save_pickle(eval_labels_pickle, eval_labels)

        print(f"Train data shape: {train_data.shape}")  # Should be (num_samples, 5, 64)
        print(f"Eval files: {len(eval_files)}, Eval labels: {len(eval_labels)}")

        # Check data shape
        print(f"Train data actual shape: {train_data.shape}")
        print(f"Expected shape: (N, {param['feature']['frames']}, {param['feature']['n_mels']})")

        # Reshape if necessary
        if len(train_data.shape) == 2:
            if train_data.shape[1] == param['feature']['frames'] * param['feature']['n_mels']:
                print("Reshaping data to correct format...")
                train_data = train_data.reshape(-1, param['feature']['frames'], param['feature']['n_mels'])
                print(f"New shape: {train_data.shape}")
            else:
                logger.error(f"Cannot reshape data: {train_data.shape[1]} != {param['feature']['frames'] * param['feature']['n_mels']}")
                continue  # Skip to next directory

        #debug
        log_memory_usage("after data loading")
        #debug
        log_memory_usage("before model creation")
        # model training
        print("============== MODEL TRAINING ==============")
        model = transformer_model(
            input_shape=(
                param["feature"]["frames"], 
                param["feature"]["n_mels"]
            ),
            head_size=param["transformer"]["head_size"],
            num_heads=param["transformer"]["num_heads"],
            ff_dim=param["transformer"]["ff_dim"],
            num_transformer_blocks=param["transformer"]["num_transformer_blocks"],
            mlp_units=param["transformer"]["mlp_units"],
            dropout=param["transformer"]["dropout"]
        )
        model.summary()
        #debug
        log_memory_usage("after model creation")
        log_memory_usage("before training")

        try:
            history = compile_and_train_model_efficiently(
                model=model, 
                train_data=train_data, 
                param=param, 
                visualizer=visualizer, 
                history_img=history_img, 
                model_file=model_file
            )
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            # Try to load model if training failed but model exists
            if os.path.exists(model_file):
                logger.info(f"Loading existing model weights from {model_file}")
                model.load_weights(model_file)
            else:
                logger.error(f"No model weights available at {model_file}")
                model.summary()
        
        #debug
        log_memory_usage("after training")


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

        #debug        
        log_memory_usage("after model prediction")

        # evaluation
        print("============== EVALUATION ==============")
        y_pred = [0. for _ in eval_labels]
        y_true = eval_labels
        original_specs = []
        reconstructed_specs = []


        for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
            try:
                data = file_to_vector_array(file_name,
                                           n_mels=param["feature"]["n_mels"],
                                           frames=param["feature"]["frames"],
                                           n_fft=param["feature"]["n_fft"],
                                           hop_length=param["feature"]["hop_length"],
                                           power=param["feature"]["power"])
                                           
                if data.shape[0] == 0:
                    logger.warning(f"No valid features extracted from file: {file_name}")
                    continue
                    
                # Batch prediction to avoid memory spikes
                batch_size = 128
                error_list = []
                for i in range(0, len(data), batch_size):
                    batch_data = data[i:i+batch_size]
                    batch_pred = model.predict(batch_data, verbose=0)
                    batch_error = np.mean(np.square(batch_data - batch_pred), axis=(1, 2))
                    error_list.extend(batch_error)
                    
                # Use mean squared error as anomaly score
                if error_list:
                    y_pred[num] = np.mean(error_list)
                    
                    # Store only a subset of spectrograms to avoid memory issues
                    if len(original_specs) < 20:  # Limit number of stored spectrograms
                        original_specs.append(data[:10])  # Store only first 10 samples
                        reconstructed_specs.append(batch_pred[:10])

            except Exception as e:
                logger.warning(f"Error processing file: {file_name}, error: {e}")
                continue

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
            

            # Average SSIM
            ssim_scores = []
            for orig, recon in zip(original_specs, reconstructed_specs):
                for i in range(min(len(orig), len(recon))):
                    try:
                        # Check time dimension (frames) instead of Mel bands
                        if orig[i].shape[0] < param["feature"]["frames"] or recon[i].shape[0] < param["feature"]["frames"]:
                            logger.warning(f"Skipping SSIM: Insufficient frames in sample {i}")
                            continue

                        # Slice time axis (axis=0) and transpose to (n_mels, frames)
                        orig_spec = orig[i][:param["feature"]["frames"], :].T  # Shape: (n_mels, frames)
                        recon_spec = recon[i][:param["feature"]["frames"], :].T

                        # Validate shapes (should be (64, 5))
                        if orig_spec.shape != (param["feature"]["n_mels"], param["feature"]["frames"]):
                            logger.warning(f"SSIM skipped: Shape mismatch {orig_spec.shape}")
                            continue

                        # Dynamic window size (fixed for small spectrograms)
                        win_size = min(orig_spec.shape)  # min(64, 5) = 5
                        win_size = min(3, min(orig_spec.shape) - 2)
                        if win_size % 2 == 0:
                            win_size += 1

                        # Calculate SSIM
                        ssim_value = ssim(
                            orig_spec,
                            recon_spec,
                            win_size=win_size,
                            data_range=orig_spec.max() - orig_spec.min() + 1e-10
                        )
                        ssim_scores.append(ssim_value)

                    except Exception as e:
                        logger.warning(f"SSIM error: {e}")

            # Save average SSIM
            evaluation_result["SSIM"] = float(np.mean(ssim_scores) if ssim_scores else -1.0)
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")

        results[evaluation_result_key] = evaluation_result
        
        # Clear memory at the end of each machine iteration
        tf.keras.backend.clear_session()
        gc.collect()

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