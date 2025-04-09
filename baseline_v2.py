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
# from import
from tqdm import tqdm
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
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
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
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
                                           power=power)
        
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
def keras_model(input_dim):
    """
    define the keras model
    the model based on the simple dense auto encoder (64*64*8*64*64)
    """
    inputLayer = Input(shape=(input_dim,))
    h = Dense(64, activation="relu")(inputLayer)
    h = Dense(64, activation="relu")(h)
    h = Dense(8, activation="relu")(h)
    h = Dense(64, activation="relu")(h)
    h = Dense(64, activation="relu")(h)
    h = Dense(input_dim, activation=None)(h)

    return Model(inputs=inputLayer, outputs=h)


########################################################################


########################################################################
# main
########################################################################
def main():
    
    # Record the start time
    start_time = time.time()

    # load parameter yaml
    with open("baseline.yaml", "r") as stream:
        param = yaml.safe_load(stream)

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

        # model training
        print("============== MODEL TRAINING ==============")
        model = keras_model(param["feature"]["n_mels"] * param["feature"]["frames"])
        model.summary()

        # training
        if os.path.exists(model_file):
            model.load_weights(model_file)
            logger.info("Model loaded from file, no training performed")
        else:

            # Update compile parameters for TensorFlow 2.x
            compile_params = param["fit"]["compile"].copy()
            if "optimizer" in compile_params and compile_params["optimizer"] == "adam":
                compile_params["optimizer"] = tf.keras.optimizers.Adam()
                
            model.compile(**compile_params)
            
            # Use TensorFlow callbacks
            callbacks = []
            if param["fit"].get("early_stopping", False):
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ))
                
            history = model.fit(
                train_data,
                train_data,
                epochs=param["fit"]["epochs"],
                batch_size=param["fit"]["batch_size"],
                shuffle=param["fit"]["shuffle"],
                validation_split=param["fit"]["validation_split"],
                verbose=param["fit"]["verbose"],
                callbacks=callbacks
            )

            visualizer.loss_plot(history.history["loss"], history.history["val_loss"])
            visualizer.save_figure(history_img)
            model.save_weights(model_file)

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
                    
                pred = model.predict(data, verbose=0)
                error = np.mean(np.square(data - pred), axis=1)
                y_pred[num] = np.mean(error)
                original_specs.append(data)
                reconstructed_specs.append(pred)
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
            

            # Average SSIM
            ssim_scores = []
            for orig, recon in zip(original_specs, reconstructed_specs):
                for i in range(min(len(orig), len(recon))):
                    try:
                        # Reshape vectors back to spectrograms if needed
                        orig_spec = orig[i].reshape(param["feature"]["n_mels"], param["feature"]["frames"])
                        recon_spec = recon[i].reshape(param["feature"]["n_mels"], param["feature"]["frames"])
                        
                        # Get dimensions
                        height, width = orig_spec.shape
                        
                        # Calculate appropriate window size (must be odd and smaller than image)
                        win_size = min(height, width)
                        if win_size % 2 == 0:  # Make odd if even
                            win_size -= 1
                        if win_size > 1:  # Ensure at least 3 for ssim
                            win_size = max(3, win_size)
                        else:
                            # Skip if dimensions are too small
                            continue
                            
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
                        
            if ssim_scores:
                evaluation_result["SSIM"] = float(np.mean(ssim_scores))
            else:
                evaluation_result["SSIM"] = float(-1)  # Indicate no valid SSIM could be calculated
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")

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