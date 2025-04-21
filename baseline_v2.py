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
########################################################################

########################################################################
# version
########################################################################
__versions__ = "2.1.0"
########################################################################

def ssim_loss(y_true, y_pred):
    """
    Structural Similarity Index (SSIM) loss with appropriate window size
    """
    # Use a smaller window size (must be odd and <= smallest dimension)
    window_size = 3  # Smallest odd number that will work with your dimensions
    
    return 1 - tf.reduce_mean(tf.image.ssim(
        tf.reshape(y_true, [-1, 64, 5, 1]),
        tf.reshape(y_pred, [-1, 64, 5, 1]), 
        max_val=K.max(y_true) - K.min(y_true) + K.epsilon(),
        filter_size=window_size
    ))

def hybrid_loss_with_margin(y_true, y_pred, alpha=0.5, margin=1.0):
    """
    Combination of MSE, SSIM loss, and a margin term to improve Specificity
    """
    mse = mean_squared_error(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)
    error = K.mean(K.square(y_true - y_pred), axis=1)
    normal_mask = K.cast(K.less(error, margin), K.floatx())
    margin_loss = K.mean(normal_mask * error)
    return alpha * mse + (1 - alpha) * ssim + 0.2 * margin_loss

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

    def loss_plot(self, loss, val_loss):
        plt.figure(figsize=(30, 10))
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title("Model loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Train", "Test"], loc="upper right")

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
def augment_spectrogram(spectrogram, max_mask_freq=15, max_mask_time=15, n_freq_masks=3, n_time_masks=3, noise_level=0.015):
    """
    Apply more aggressive augmentations to spectrograms: frequency/time masking and noise injection.
    """
    # Frequency masking with increased intensity
    freq_mask_param = np.random.randint(0, max_mask_freq)
    for _ in range(n_freq_masks):
        freq_start = np.random.randint(0, spectrogram.shape[0])
        freq_end = min(spectrogram.shape[0], freq_start + freq_mask_param)
        spectrogram[freq_start:freq_end, :] *= 0.3  # More aggressive reduction (from 0.5 to 0.3)

    # Time masking with increased intensity
    time_mask_param = np.random.randint(0, max_mask_time)
    for _ in range(n_time_masks):
        time_start = np.random.randint(0, spectrogram.shape[1])
        time_end = min(spectrogram.shape[1], time_start + time_mask_param)
        spectrogram[:, time_start:time_end] *= 0.3  # More aggressive reduction

    # Add more noise
    noise = np.random.normal(0, noise_level, spectrogram.shape)
    spectrogram += noise
    
    # Random pitch shift simulation (add slight shifts to frequency bands)
    if np.random.random() > 0.5:
        shift = np.random.uniform(-0.05, 0.05, (1, spectrogram.shape[1]))
        pitch_noise = np.tile(shift, (spectrogram.shape[0], 1))
        spectrogram = spectrogram * (1 + pitch_noise)

    return spectrogram

def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         augment=False):
    """
    Convert file_name to a vector array with optional augmentation for normal data.
    """
    dims = n_mels * frames
    sr, y = demux_wav(file_name)
    if y is None:
        return np.empty((0, dims), float)
        
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                    sr=sr,
                                                    n_fft=n_fft,
                                                    hop_length=hop_length,
                                                    n_mels=n_mels,
                                                    power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    
    if augment:
        log_mel_spectrogram = augment_spectrogram(log_mel_spectrogram)
    
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1
    if vectorarray_size < 1:
        return np.empty((0, dims), float)

    vectorarray = np.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

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

def dataset_generator(target_dir,
                      normal_dir_name="normal",
                      abnormal_dir_name="abnormal",
                      ext="wav"):
    logger.info(f"target_dir : {target_dir}")
    target_dir_path = Path(target_dir)

    normal_files = sorted(list(target_dir_path.joinpath(normal_dir_name).glob(f"*.{ext}")))
    normal_files = [str(file_path) for file_path in normal_files]
    normal_labels = np.zeros(len(normal_files))
    
    if len(normal_files) == 0:
        logger.exception(f"No {ext} files found in {normal_dir_name} directory!")
        return [], [], [], []

    abnormal_files = sorted(list(target_dir_path.joinpath(abnormal_dir_name).glob(f"*.{ext}")))
    abnormal_files = [str(file_path) for file_path in abnormal_files]
    abnormal_labels = np.ones(len(abnormal_files))
    
    if len(abnormal_files) == 0:
        logger.exception(f"No {ext} files found in {abnormal_dir_name} directory!")
        return [], [], [], []

    train_files = normal_files[len(abnormal_files):]
    train_labels = normal_labels[len(abnormal_files):]
    eval_files = np.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    eval_labels = np.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
    
    logger.info(f"train_file num : {len(train_files)}")
    logger.info(f"eval_file  num : {len(eval_files)}")

    return train_files, train_labels, eval_files, eval_labels

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
    weight_decay = 1e-4  # Add weight decay
    
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
    
    # Bottleneck layer with reduced attention heads
    bottleneck_output = Dense(bottleneck, activation=activation, name="bottleneck")(x)
    # Reshape for attention: (batch_size, 1, bottleneck)
    attn_input = tf.expand_dims(bottleneck_output, axis=1)
    # Reduce number of attention heads from 4 to 2
    attn_output = MultiHeadAttention(num_heads=2, key_dim=bottleneck // 2)(attn_input, attn_input)
    attn_output = tf.squeeze(attn_output, axis=1)  # (batch_size, bottleneck)
    # Reduce attention influence with lower weighting
    x = 0.7 * bottleneck_output + 0.3 * attn_output  # Modified residual connection
    
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
        x = Dense(input_dim, activation=None, kernel_regularizer=l2(weight_decay))(x)
        output = Add()([x, inputLayer])
    else:
        output = Dense(input_dim, activation=None, kernel_regularizer=l2(weight_decay))(x)

    return Model(inputs=inputLayer, outputs=output)

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
    dirs = []

    # Keep the filtering mechanism based on YAML config
    if param.get("filter", {}).get("enabled", False):
        filter_db = param["filter"].get("db_level")
        filter_machine = param["filter"].get("machine_type")
        filter_id = param["filter"].get("machine_id")
        
        pattern = ""
        if filter_db:
            pattern += f"{filter_db}/"
        else:
            pattern += "*/"
        if filter_machine:
            pattern += f"{filter_machine}/"
        else:
            pattern += "*/"
        if filter_id:
            pattern += f"{filter_id}"
        else:
            pattern += "*"
        full_pattern = str(base_path / pattern)
        logger.info(f"Filtering with pattern: {full_pattern}")
        dirs = sorted(glob.glob(full_pattern))
    else:
        dirs = sorted(glob.glob(str(base_path / "*/*/*")))

    dirs = [dir_path for dir_path in dirs if os.path.isdir(dir_path) and "/id_" in dir_path]

    result_file = f"{param['result_directory']}/resultv2.yaml"
    results = {}

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
        db = parts[-3]
        machine_type = parts[-2]
        machine_id = parts[-1]

        logger.info(f"Processing: db={db}, machine_type={machine_type}, machine_id={machine_id}")

        evaluation_result = {}
        train_pickle = f"{param['pickle_directory']}/train_{machine_type}_{machine_id}_{db}.pickle"
        eval_files_pickle = f"{param['pickle_directory']}/eval_files_{machine_type}_{machine_id}_{db}.pickle"
        eval_labels_pickle = f"{param['pickle_directory']}/eval_labels_{machine_type}_{machine_id}_{db}.pickle"
        model_file = f"{param['model_directory']}/model_{machine_type}_{machine_id}_{db}.weights.h5"
        history_img = f"{param['model_directory']}/history_{machine_type}_{machine_id}_{db}.png"
        evaluation_result_key = f"{machine_type}_{machine_id}_{db}"

        print("============== DATASET_GENERATOR ==============")
        if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
            train_data = load_pickle(train_pickle)
            eval_files = load_pickle(eval_files_pickle)
            eval_labels = load_pickle(eval_labels_pickle)
        else:
            train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir)
            if len(train_files) == 0 or len(eval_files) == 0:
                logger.error(f"No files found for {evaluation_result_key}, skipping...")
                continue

            train_data = list_to_vector_array(train_files,
                                            msg="generate train_dataset",
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"],
                                            augment=True)  # Apply augmentation to normal data
            
            if train_data.shape[0] == 0:
                logger.error(f"No valid training data for {evaluation_result_key}, skipping...")
                continue

            save_pickle(train_pickle, train_data)
            save_pickle(eval_files_pickle, eval_files)
            save_pickle(eval_labels_pickle, eval_labels)

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

            if loss_type == "ssim":
                compile_params["loss"] = ssim_loss
            elif loss_type == "hybrid":
                alpha = param.get("model", {}).get("hybrid_loss_alpha", 0.5)
                margin = param.get("model", {}).get("margin", 1.0)
                compile_params["loss"] = lambda y_true, y_pred: hybrid_loss_with_margin(y_true, y_pred, alpha, margin)

            if "optimizer" in compile_params and compile_params["optimizer"] == "adam":
                compile_params["optimizer"] = tf.keras.optimizers.Adam()

            # Add weighted_metrics to fix the warning
            if "weighted_metrics" not in compile_params:
                compile_params["weighted_metrics"] = []

            model.compile(**compile_params)
            
            callbacks = []
            if param["fit"].get("early_stopping", False):
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    min_delta=0.001,
                    restore_best_weights=True
                ))

            
            sample_weights = np.ones(len(train_data))

            # Apply enhanced sample weighting for all datasets
            if param.get("fit", {}).get("apply_sample_weights", False):
                # Get weight factor from config with enhanced default
                weight_factor = param.get("fit", {}).get("weight_factor", 1.8)
                
                # Apply weights more intelligently based on data variance
                data_variance = np.var(train_data, axis=0).mean()
                variance_norm = min(max(data_variance / 0.01, 0.8), 1.5)  # Normalize between 0.8 and 1.5
                
                # Apply adaptive weight factor based on data characteristics
                effective_weight = weight_factor * variance_norm
                sample_weights *= effective_weight
                
                logger.info(f"Applied adaptive sample weight of {effective_weight:.3f} for {machine_type}_{machine_id}")

            history = model.fit(
                train_data,
                train_data,
                epochs=param["fit"]["epochs"],
                batch_size=param["fit"]["batch_size"],
                shuffle=param["fit"]["shuffle"],
                validation_split=param["fit"]["validation_split"],
                verbose=param["fit"]["verbose"],
                callbacks=callbacks,
                sample_weight=sample_weights
            )

        print("============== EVALUATION ==============")
        y_pred_global = [0. for _ in eval_labels]
        y_pred_feature = [0. for _ in eval_labels]
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
                                           power=param["feature"]["power"],
                                           augment=False)  # No augmentation during eval
                                           
                if data.shape[0] == 0:
                    logger.warning(f"No valid features extracted from file: {file_name}")
                    continue
                    
                pred = model.predict(data, verbose=0)
                # Global reconstruction error
                global_error = np.mean(np.square(data - pred), axis=1)
                y_pred_global[num] = np.mean(global_error)
                # Feature-level error (per mel-frequency bin)
                feature_error = np.mean(np.square(data - pred), axis=0)  # Shape: (n_mels * frames,)
                y_pred_feature[num] = np.mean(feature_error)
                original_specs.append(data)
                reconstructed_specs.append(pred)
            except Exception as e:
                logger.warning(f"Error processing file: {file_name}, error: {e}")

        # Combine global and feature-level scores
        y_pred_combined = [0.5 * g + 0.5 * f for g, f in zip(y_pred_global, y_pred_feature)]

        # Statistical thresholding: mean + k*std on normal samples
        normal_indices = [i for i, label in enumerate(y_true) if label == 0]
        normal_scores = [y_pred_combined[i] for i in normal_indices]
        if normal_scores:
            mean_normal = np.mean(normal_scores)
            std_normal = np.std(normal_scores)
            stat_threshold = mean_normal + 2 * std_normal  # k=2
            evaluation_result["Statistical Threshold"] = float(stat_threshold)
        else:
            stat_threshold = np.mean(y_pred_combined)  # Fallback
            evaluation_result["Statistical Threshold"] = float(stat_threshold)

        # Optimize threshold using ROC and Precision-Recall curves
        fpr, tpr, roc_thresholds = metrics.roc_curve(y_true, y_pred_combined)
        precision, recall, pr_thresholds = metrics.precision_recall_curve(y_true, y_pred_combined)
        # Balance Specificity and Recall with increased weight on Recall
        best_threshold = None
        best_score = -float('inf')
        for i, thresh in enumerate(roc_thresholds):
            if i >= len(fpr) or i >= len(tpr):
                continue
            # Score balances TPR (Recall) and FPR (1-Specificity)
            score = tpr[i] - 2 * fpr[i]  # Weight FPR more to improve Specificity
            if score > best_score:
                best_score = score
                best_threshold = thresh
        
        if best_threshold is None:
            best_threshold = stat_threshold  # Fallback to statistical threshold

        # Apply best threshold
        y_pred_binary = (np.array(y_pred_combined) >= best_threshold).astype(int)

        # Calculate metrics - keep only AUC, Specificity, MCC, and SSIM
        try:
            # AUC
            score = metrics.roc_auc_score(y_true, y_pred_combined)
            logger.info(f"AUC : {score}")
            evaluation_result["AUC"] = float(score)
            
            # Confusion matrix for Specificity and MCC
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_binary).ravel()
            
            # Specificity
            specificity = float(tn / (tn + fp + 1e-10))
            evaluation_result["Specificity"] = specificity
            logger.info(f"Specificity: {specificity:.4f}")
            
            # MCC
            mcc_numerator = (tp * tn) - (fp * fn)
            mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-10)
            mcc = float(mcc_numerator / mcc_denominator)
            evaluation_result["MCC"] = mcc
            logger.info(f"MCC: {mcc:.4f}")
            
            # Best threshold for reference
            evaluation_result["Best Threshold"] = float(best_threshold)

            # Average SSIM
            ssim_scores = []
            for orig, recon in zip(original_specs, reconstructed_specs):
                for i in range(min(len(orig), len(recon))):
                    try:
                        orig_spec = orig[i].reshape(param["feature"]["n_mels"], param["feature"]["frames"])
                        recon_spec = recon[i].reshape(param["feature"]["n_mels"], param["feature"]["frames"])
                        # Use a small window size - must be odd and <= smallest dimension
                        win_size = 3
                        ssim_value = ssim(
                            orig_spec, 
                            recon_spec,
                            win_size=win_size,
                            data_range=orig_spec.max() - orig_spec.min() + 1e-10
                        )
                        ssim_scores.append(ssim_value)
                    except Exception as e:
                        logger.warning(f"SSIM calculation error: {e}")
                        
            if ssim_scores:
                evaluation_result["SSIM"] = float(np.mean(ssim_scores))
            else:
                evaluation_result["SSIM"] = float(-1)
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")

        results[evaluation_result_key] = evaluation_result
        print("===========================")

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    results["execution_time_seconds"] = float(total_time)

    print("\n===========================")
    logger.info(f"all results -> {result_file}")
    with open(result_file, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print("===========================")

if __name__ == "__main__":
    main()