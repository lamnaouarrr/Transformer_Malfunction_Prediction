#!/usr/bin/env python
"""
 @file   anomaly_detection_api.py
 @brief  API for WAV file analysis using the FNN model for anomaly detection (FastAPI version)
"""
import os
import sys
import numpy as np
import yaml
import logging
import tempfile
import librosa
import librosa.core
import librosa.feature
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, BatchNormalization, Add
from tensorflow.keras.regularizers import l2

# Set up the system path to ensure imports work correctly
# Assumes the script is run from the 'app' directory or the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__)) # Add current script directory

# Apply TensorFlow patch before any imports that use TensorFlow
try:
    # Assuming tensorflow_patch.py is in the same directory or accessible via sys.path
    import tensorflow_patch
    tensorflow_patch.apply_patch()
    print("✓ Applied patch for tf.keras.losses.mean_squared_error")
except ImportError as e:
    print(f"Warning: Could not import tensorflow_patch: {e}. Proceeding without patch.")
except Exception as e:
    print(f"Error applying tensorflow_patch: {e}")
    sys.exit(1)


# Now import TensorFlow and verify the patch if applied
import tensorflow as tf
# Check if patch was intended and failed
if 'tensorflow_patch' in sys.modules and not hasattr(tf.keras.losses, 'mean_squared_error'):
    print("Patch failed: mean_squared_error not found in tf.keras.losses after attempting patch.")
    # Decide if this is critical; exiting might be appropriate
    # sys.exit(1)
elif not hasattr(tf.keras.losses, 'mean_squared_error'):
     print("Warning: mean_squared_error not found in tf.keras.losses (patch might be needed or TF version mismatch).")


# Define setup_logging first
def setup_logging():
    """Set up logging for the application"""
    log_dir = os.path.join(project_root, "logs", "log_fnn")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "baseline_fnn.log")
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) # Also log to console
        ]
    )
    # Reduce verbosity of libraries if needed
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.ERROR) # Suppress TF info messages
    
    logger = logging.getLogger(__name__) # Use module name for logger
    logger.info("Logging configured.")
    return logger

# Then call it
logger = setup_logging()


# Load configuration
yaml_path = os.path.join(project_root, 'baseline_fnn.yaml')
try:
    with open(yaml_path, "r") as stream:
        param = yaml.safe_load(stream)
    logger.info(f"Configuration loaded from {yaml_path}")
except FileNotFoundError:
    logger.error(f"Configuration file not found: {yaml_path}")
    sys.exit(1)
except yaml.YAMLError as e:
    logger.error(f"Error parsing configuration file {yaml_path}: {e}")
    sys.exit(1)


# Initialize FastAPI app
app = FastAPI(
    title="Anomaly Detection API",
    description="API for WAV file analysis using the FNN model for anomaly detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# Global variables
model = None
model_file_path = None # Store the actual path used
threshold = 0.5  # Default threshold


def file_load(wav_name):
    """Load a wav file."""
    try:
        # Load audio file with specified sample rate, ensure mono
        y, sr = librosa.load(wav_name, sr=param["feature"]["sr"], mono=True)
        logger.debug(f"Loaded {wav_name}, sr={sr}, length={len(y)}")
        return y, sr
    except Exception as e:
        logger.error(f"Error loading {wav_name}: {e}")
        # Return None for both values to indicate failure
        return None, None


def demux_wav(wav_name, channel=0):
    """
    Demuxes a WAV file to get a specific channel or converts stereo to mono.
    Note: librosa.load with mono=True handles mono conversion.
          This function might be simplified or removed if only mono is needed.
    """
    try:
        # Use librosa.load with mono=False to potentially get multi-channel data
        multi_channel_data, sr = librosa.load(wav_name, sr=param["feature"]["sr"], mono=False)
        if multi_channel_data is None:
            logger.error(f"Failed to load {wav_name} for demuxing.")
            return None, None

        if multi_channel_data.ndim == 1:
            # Already mono
            logger.debug(f"Audio {wav_name} is already mono.")
            return sr, multi_channel_data
        elif multi_channel_data.ndim > 1 and multi_channel_data.shape[0] > channel:
            # Select the specified channel
            logger.debug(f"Extracting channel {channel} from {wav_name}.")
            return sr, multi_channel_data[channel, :]
        else:
            logger.warning(f"Channel {channel} not available in {wav_name} (shape: {multi_channel_data.shape}). Falling back to mono conversion.")
            # Fallback to mono using librosa's method
            y_mono, sr_mono = librosa.load(wav_name, sr=param["feature"]["sr"], mono=True)
            return sr_mono, y_mono

    except Exception as e:
        logger.error(f"Error in demux_wav for {wav_name}: {e}")
        return None, None


def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    Convert audio file to overlapping frames of log-Mel spectrogram features.
    """
    dims = n_mels * frames
    
    try:
        # Load audio using file_load which ensures mono and correct sr
        y, sr = file_load(file_name)
        if y is None:
            logger.error(f"Failed to load audio from {file_name}")
            return np.empty((0, dims), float)

        # Check minimum length required for one FFT window
        if len(y) < n_fft:
            logger.warning(f"File too short for STFT: {file_name} (length {len(y)} < n_fft {n_fft})")
            return np.empty((0, dims), float)

        # Compute Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                         sr=sr,
                                                         n_fft=n_fft,
                                                         hop_length=hop_length,
                                                         n_mels=n_mels,
                                                         power=power)

        # Convert to log-Mel spectrogram (dB scale)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        logger.debug(f"Log Mel Spectrogram shape: {log_mel_spectrogram.shape} for {file_name}")

        # Check if enough frames for the desired frame aggregation
        if log_mel_spectrogram.shape[1] < frames:
            logger.warning(f"Not enough frames ({log_mel_spectrogram.shape[1]}) "
                           f"to create feature vectors of size {frames} in {file_name}")
            return np.empty((0, dims), float)

        # Create framed feature vectors efficiently
        vectorarray_size = log_mel_spectrogram.shape[1] - frames + 1
        vectorarray = np.zeros((vectorarray_size, dims), float)
        for t in range(frames):
            vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

        logger.info(f"Extracted features shape: {vectorarray.shape} for {file_name}")
        return vectorarray

    except Exception as e:
        logger.error(f"Error in file_to_vector_array for {file_name}: {e}", exc_info=True)
        return np.empty((0, dims), float)


def keras_model(input_dim, config=None):
    """
    Define the Keras Autoencoder model based on configuration.
    """
    if config is None:
        config = {}
    logger.info(f"Building Keras model with config: {config}")

    # Extract parameters from config with defaults
    depth = config.get("depth", 3)
    width = config.get("width", 128) # Default width for first dense layer
    bottleneck = config.get("bottleneck", 16) # Default bottleneck size
    dropout_rate = config.get("dropout", 0.1) # Default dropout
    use_batch_norm = config.get("batch_norm", True) # Default BN
    use_residual = config.get("residual", False) # Default residual connections
    activation = config.get("activation", "relu") # Default activation
    weight_decay = config.get("weight_decay", 1e-5) # Default weight decay

    inputLayer = Input(shape=(input_dim,), name="input_layer")
    x = inputLayer

    # --- Encoder ---
    logger.debug("Building Encoder")
    encoder_layers = [] # To potentially use for residual connections
    for i in range(depth):
        layer_width = max(width // (2 ** i), bottleneck) if i < depth -1 else bottleneck
        layer_input = x # Store input for potential residual connection
        logger.debug(f" Encoder Layer {i+1}: width={layer_width}, input_shape={x.shape}")

        x = Dense(layer_width, activation=None, kernel_regularizer=l2(weight_decay), name=f"enc_dense_{i}")(x)
        if use_batch_norm:
            x = BatchNormalization(name=f"enc_bn_{i}")(x)
        x = Activation(activation, name=f"enc_act_{i}")(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name=f"enc_dropout_{i}")(x)

        # Residual Connection (if enabled and dimensions match)
        if use_residual and layer_input.shape[-1] == layer_width and i > 0: # Avoid residual on first layer unless specifically designed
             x = Add(name=f"enc_residual_{i}")([x, layer_input])
             logger.debug(f"  Added residual connection for Encoder Layer {i+1}")
        encoder_layers.append(x)

    bottleneck_output = x # Last layer of encoder is the bottleneck
    logger.info(f"Bottleneck layer shape: {bottleneck_output.shape}")

    # --- Decoder ---
    logger.debug("Building Decoder")
    for i in range(depth - 1, -1, -1): # Iterate backwards from depth-1 down to 0
        # Determine the target width for this decoder layer (mirroring encoder)
        is_output_layer = (i == 0)
        if is_output_layer:
            layer_width = input_dim # Final layer must match input dimension
        else:
            # Target the width of the corresponding encoder layer's *input*
            # This requires careful indexing or storing encoder shapes
            # Simpler approach: Mirror the widths used in encoder construction
             prev_encoder_layer_index = i -1
             if prev_encoder_layer_index >= 0:
                 # Width before the corresponding encoder layer activation
                 layer_width = max(width // (2**prev_encoder_layer_index), bottleneck) if prev_encoder_layer_index < depth-1 else bottleneck
             else:
                 # The layer that should map back towards the first dense layer width
                 layer_width = width # Target the width of the first main dense layer


        layer_input = x # Store input for potential residual connection
        logger.debug(f" Decoder Layer {depth-i}: target_width={layer_width}, input_shape={x.shape}")

        # Use 'linear' activation for the final output layer, otherwise configured activation
        final_activation = None if is_output_layer else activation

        x = Dense(layer_width, activation=None, kernel_regularizer=l2(weight_decay), name=f"dec_dense_{depth-1-i}")(x)
        if use_batch_norm and not is_output_layer: # Often BN is skipped before output
            x = BatchNormalization(name=f"dec_bn_{depth-1-i}")(x)
        if final_activation: # Apply activation if not the output layer
             x = Activation(final_activation, name=f"dec_act_{depth-1-i}")(x)
        if dropout_rate > 0 and not is_output_layer: # Usually no dropout on output
            x = Dropout(dropout_rate, name=f"dec_dropout_{depth-1-i}")(x)

        # Residual Connection (if enabled and dimensions match)
        # Note: Residuals in decoders are less common or need careful design (e.g., skip connections from encoder)
        # Simple residual add like in encoder might not be appropriate here. Sticking to encoder residuals for now.
        # if use_residual and layer_input.shape[-1] == layer_width and not is_output_layer:
        #     x = Add(name=f"dec_residual_{depth-1-i}")([x, layer_input])
        #     logger.debug(f"  Added residual connection for Decoder Layer {depth-i}")


    outputLayer = x # Final output of the decoder
    logger.info(f"Output layer shape: {outputLayer.shape}")

    # Define the model
    model = Model(inputs=inputLayer, outputs=outputLayer, name="FNN_Autoencoder")

    # Compile with Mean Squared Error loss for reconstruction
    optimizer = tf.keras.optimizers.Adam(learning_rate=param.get("train", {}).get("learning_rate", 0.001))
    model.compile(optimizer=optimizer, loss=tf.keras.losses.mean_squared_error)
    logger.info("Model compiled successfully with MSE loss and Adam optimizer.")

    model.summary(print_fn=logger.info) # Log model summary

    return model


def normalize_features(features, method="minmax_local"):
    """
    Normalize feature vectors.
    'minmax_local': Normalize each feature vector (frame) independently to [0, 1].
    'minmax_global': Normalize across the entire dataset stats (requires pre-calculation).
    'zscore_local': Standardize each feature vector independently (mean=0, std=1).
    'zscore_global': Standardize across the entire dataset stats (requires pre-calculation).
    """
    if features.size == 0:
        return features

    logger.debug(f"Normalizing features with method: {method}")

    if method == "minmax_local":
        min_vals = np.min(features, axis=1, keepdims=True)
        max_vals = np.max(features, axis=1, keepdims=True)
        scale = max_vals - min_vals
        scale[scale == 0] = 1 # Avoid division by zero for constant vectors
        return (features - min_vals) / scale
    elif method == "zscore_local":
        mean_vals = np.mean(features, axis=1, keepdims=True)
        std_vals = np.std(features, axis=1, keepdims=True)
        std_vals[std_vals == 0] = 1 # Avoid division by zero
        return (features - mean_vals) / std_vals
    # Add global methods here if mean/std/min/max are pre-calculated and loaded
    # elif method == "minmax_global":
    #     # Requires global_min, global_max
    #     pass
    # elif method == "zscore_global":
    #     # Requires global_mean, global_std
    #     pass
    elif method == "none":
         logger.debug("No normalization applied.")
         return features
    else:
        logger.warning(f"Unknown normalization method: {method}. Returning original features.")
        return features


def load_trained_model():
    """Load the trained autoencoder model and optimal threshold."""
    global model, model_file_path, threshold, param # Make param accessible

    # Construct the absolute path to the model file relative to the project root
    model_filename = param.get("model_directory", "model/FNN/model_overall.h5")
    model_file_path = os.path.join(project_root, model_filename.replace('./', '')) # Ensure relative paths work

    logger.info(f"Attempting to load model from: {model_file_path}")

    if not os.path.exists(model_file_path):
        logger.error(f"Model file not found: {model_file_path}")
        return False

    try:
        # Create a new model instance based on config
        model_config = param.get("model", {}).get("architecture", {})
        input_dim = param["feature"]["n_mels"] * param["feature"]["frames"]
        model = keras_model(input_dim, config=model_config)

        # Load the weights into the newly created model structure
        model.load_weights(model_file_path)
        logger.info(f"Successfully loaded model weights from: {model_file_path}")

        # Try to load the optimal threshold from the result file
        result_filename = param.get("result_file", "results/result_fnn.yaml")
        result_file_path = os.path.join(project_root, result_filename.replace('./', ''))

        if os.path.exists(result_file_path):
            try:
                with open(result_file_path, "r") as f:
                    results = yaml.safe_load(f)
                    # Adjust key based on actual content of result_fnn.yaml
                    # Common keys might be 'AUC_ROC', 'pAUC', 'threshold', 'optimal_threshold'
                    # Assuming the key is 'decision_threshold' or 'optimal_threshold'
                    loaded_threshold = results.get("decision_threshold", results.get("optimal_threshold"))
                    if loaded_threshold is not None:
                         threshold = float(loaded_threshold) # Ensure it's a float
                         logger.info(f"Loaded optimal threshold from {result_file_path}: {threshold}")
                    else:
                         logger.warning(f"Could not find 'decision_threshold' or 'optimal_threshold' in {result_file_path}. Using default: {threshold}")
            except yaml.YAMLError as e:
                 logger.warning(f"Error parsing result file {result_file_path}: {e}. Using default threshold: {threshold}")
            except Exception as e:
                 logger.warning(f"Could not read or process result file {result_file_path}: {e}. Using default threshold: {threshold}")
        else:
            logger.warning(f"Result file not found at {result_file_path}. Using default threshold: {threshold}")

        return True

    except Exception as e:
        logger.error(f"Error loading model or weights from {model_file_path}: {e}", exc_info=True)
        return False


def analyze_wav(wav_file_path):
    """
    Analyze a single WAV file: extract features, predict reconstruction error, and classify.

    Args:
        wav_file_path: Path to the WAV file.

    Returns:
        dict: Analysis result including prediction, anomaly score, and threshold.
              Returns {"error": ...} if an error occurs.
    """
    global model, threshold, param # Ensure access to globals

    if model is None:
         logger.error("Model is not loaded. Cannot analyze.")
         return {"error": "Model not loaded"}

    try:
        logger.info(f"Analyzing WAV file: {wav_file_path}")
        # 1. Extract features
        features = file_to_vector_array(
            wav_file_path,
            n_mels=param["feature"]["n_mels"],
            frames=param["feature"]["frames"],
            n_fft=param["feature"]["n_fft"],
            hop_length=param["feature"]["hop_length"],
            power=param["feature"]["power"]
        )

        if features.shape[0] == 0:
            logger.warning(f"No features extracted from {wav_file_path}. Cannot predict.")
            # Return a specific result or error based on requirements
            return {"error": f"Could not extract valid features from file: {os.path.basename(wav_file_path)}"}

        # 2. Normalize features (using the method defined in config, default 'none')
        norm_method = param.get("model", {}).get("normalization_method", "none")
        if norm_method != "none":
            features = normalize_features(features, method=norm_method)
            logger.debug(f"Features normalized using method: {norm_method}")


        # 3. Predict using the autoencoder model
        reconstructed_features = model.predict(features, verbose=0)
        logger.debug(f"Prediction complete. Input shape: {features.shape}, Output shape: {reconstructed_features.shape}")


        # 4. Calculate reconstruction error (Mean Squared Error per frame)
        # Ensure both are float32 for consistent calculation
        features_f32 = features.astype(np.float32)
        reconstructed_features_f32 = reconstructed_features.astype(np.float32)
        frame_errors = np.mean(np.square(features_f32 - reconstructed_features_f32), axis=1)
        logger.debug(f"Calculated frame errors shape: {frame_errors.shape}")

        # 5. Calculate the overall anomaly score for the file (e.g., mean or max error)
        # Using mean error is common for autoencoders
        anomaly_score = float(np.mean(frame_errors))
        logger.info(f"Anomaly score for {wav_file_path}: {anomaly_score:.6f}")

        # 6. Compare score with the threshold
        is_abnormal = anomaly_score >= threshold
        prediction_label = "abnormal" if is_abnormal else "normal"

        logger.info(f"File: {os.path.basename(wav_file_path)}, Score: {anomaly_score:.4f}, Threshold: {threshold:.4f}, Prediction: {prediction_label}")

        return {
            "file_name": os.path.basename(wav_file_path),
            "prediction": prediction_label,
            "anomaly_score": anomaly_score,
            "threshold": threshold
        }

    except Exception as e:
        logger.error(f"Error analyzing WAV file {wav_file_path}: {e}", exc_info=True)
        return {"error": f"Analysis failed for {os.path.basename(wav_file_path)}: {str(e)}"}


# --- FastAPI Routes ---

@app.on_event("startup")
async def startup_event():
    """FastAPI startup event: Load the model."""
    logger.info("Application startup: Loading model...")
    if not load_trained_model():
        logger.critical("Failed to load the model during startup. API might not function correctly.")
        # Depending on requirements, you might want to prevent startup
        # raise RuntimeError("Model loading failed, cannot start API.")
    else:
        logger.info("Model loaded successfully. API is ready.")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        logger.warning("Health check failed: Model not loaded.")
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load.")
    logger.debug("Health check successful.")
    return {"status": "ok", "message": "API is running and model is loaded."}


@app.post("/predict", response_model=dict)
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict if an uploaded WAV file contains normal or abnormal sound.
    """
    if model is None:
         logger.error("Prediction request failed: Model not loaded.")
         raise HTTPException(status_code=503, detail="Model not loaded, cannot process request.")

    # Basic validation
    if not file:
        logger.warning("Prediction request failed: No file provided.")
        raise HTTPException(status_code=400, detail="No file provided.")

    if not file.filename.lower().endswith('.wav'):
        logger.warning(f"Prediction request failed: Invalid file type '{file.filename}'. Only WAV files are supported.")
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}. Only WAV files are supported.")

    logger.info(f"Received prediction request for file: {file.filename}")

    # Use a temporary file to save the upload
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_path = temp_wav.name
            logger.debug(f"Saving uploaded file to temporary path: {temp_path}")
            # Read the file content chunk by chunk to handle large files
            while contents := await file.read(1024 * 1024): # Read in 1MB chunks
                 temp_wav.write(contents)
            logger.debug(f"Finished writing uploaded file to {temp_path}")

        # Ensure file is fully written before analysis
        # (with block handles closing)

        # Analyze the temporary WAV file
        analysis_result = analyze_wav(temp_path)

    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error processing file: {str(e)}")
    finally:
        # Clean up the temporary file
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.error(f"Error deleting temporary file {temp_path}: {e}")
        # Ensure the uploaded file resource is closed
        await file.close()


    # Handle analysis errors
    if "error" in analysis_result:
        logger.error(f"Analysis failed for {file.filename}: {analysis_result['error']}")
        # Return a specific error code based on the type of error if possible
        if "Could not extract valid features" in analysis_result['error']:
             raise HTTPException(status_code=400, detail=analysis_result['error'])
        else:
             raise HTTPException(status_code=500, detail=analysis_result['error'])

    logger.info(f"Prediction successful for {file.filename}")
    return analysis_result


# (Optional) Keep predict_data if needed, but ensure it aligns with analyze_wav logic
# @app.post("/predict_data") ...


# --- Main Execution ---

if __name__ == '__main__':
    logger.info("Starting FastAPI server...")
    # The model is loaded via the startup_event, no need to call load_trained_model here directly.
    # The redundant block that caused the error has been removed.

    # Get host and port from environment variables or defaults
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", 8000))

    logger.info(f"Server will run on {api_host}:{api_port}")

    # Start the FastAPI app with uvicorn
    uvicorn.run(
        "anomaly_detection_api:app", # Point to the FastAPI app instance
        host=api_host,
        port=api_port,
        reload=False, # Disable reload for production/stable runs; enable for development if needed
        log_level="info" # Control uvicorn's logging level
    )

