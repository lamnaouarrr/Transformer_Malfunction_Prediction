#!/usr/bin/env python
"""
 @file   anomaly_detection_api.py
 @brief  API for WAV file analysis using the FNN model for anomaly detection (FastAPI version)
"""
import sys
import os
import logging
import yaml
import time
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
from typing import Optional, Dict, Any, List
import uvicorn
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Set up the system path to ensure imports work correctly
# Assumes the script is run from the 'app' directory or the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__)) # Add current script directory

# Apply TensorFlow patch before any imports that use TensorFlow
try:
    import tensorflow_patch
    tensorflow_patch.apply_patch()
    print("✓ Applied patch for tf.keras.losses.mean_squared_error")
except ImportError as e:
    print(f"Warning: Could not import tensorflow_patch: {e}. Proceeding without patch.")
except Exception as e:
    print(f"Error applying tensorflow_patch: {e}")
    sys.exit(1)

# Now import TensorFlow
import tensorflow as tf

# Import the necessary libraries for feature extraction
import librosa
import librosa.core
import librosa.feature

# Define setup_logging function
def setup_logging():
    """Set up logging for the application"""
    log_dir = os.path.join(project_root, "logs", "log_fnn")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "api_debug.log")
    
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
    logger.info(f"Logging configured. Logs will be saved to {log_file}")
    return logger

# Configure logging
logger = setup_logging()

# Load configuration
yaml_path = os.path.join(project_root, 'baseline_fnn.yaml')
try:
    with open(yaml_path, "r") as stream:
        param = yaml.safe_load(stream)
    logger.info(f"Configuration loaded from {yaml_path}")
except FileNotFoundError:
    logger.error(f"Configuration file not found: {yaml_path}")
    param = {
        "feature": {"n_mels": 48, "frames": 4, "n_fft": 1024, "hop_length": 1024, "power": 2.0},
        "model": {"architecture": {"depth": 4, "width": 128, "bottleneck": 32}}
    }

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
threshold = 0.35  # Base threshold value
calibration_factor = 0.9  # Default calibration factor

def create_model(input_dim, config=None):
    """Create a new model based on configuration parameters."""
    logger.info("Creating model architecture from scratch")
    if config is None:
        config = {}
    
    depth = config.get("depth", 4)
    width = config.get("width", 128)
    bottleneck = config.get("bottleneck", 32)
    dropout_rate = config.get("dropout", 0.2)
    use_batch_norm = config.get("batch_norm", True)
    activation = config.get("activation", "sigmoid")
    weight_decay = config.get("weight_decay", 1e-4)
    
    # Create a simple autoencoder for anomaly detection
    inputLayer = Input(shape=(input_dim,))
    
    # Encoder
    x = Dense(width, kernel_regularizer=l2(weight_decay))(inputLayer)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    
    # Create deeper layers with decreasing width
    for i in range(depth - 1):
        layer_width = max(width // (2 ** (i + 1)), bottleneck)
        x = Dense(layer_width, kernel_regularizer=l2(weight_decay))(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
    
    # Bottleneck layer
    bottleneck_output = Dense(bottleneck, activation=activation, name="bottleneck")(x)
    
    # Decoder (mirror of encoder)
    x = bottleneck_output
    for i in range(depth - 1):
        layer_width = min(bottleneck * (2 ** (i + 1)), width)
        x = Dense(layer_width, kernel_regularizer=l2(weight_decay))(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
    
    # Output layer (sigmoid for binary classification)
    output = Dense(1, activation="sigmoid")(x)
    
    # Create and compile model
    model = Model(inputs=inputLayer, outputs=output)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    logger.info("Model architecture created successfully")
    return model

@tf.keras.utils.register_keras_serializable()
def binary_cross_entropy_loss(y_true, y_pred):
    """Binary cross-entropy loss for autoencoder"""
    import tensorflow.keras.backend as K
    # Normalize inputs if needed (values between 0 and 1)
    y_true_normalized = (y_true - K.min(y_true)) / (K.max(y_true) - K.min(y_true) + K.epsilon())
    y_pred_normalized = (y_pred - K.min(y_pred)) / (K.max(y_pred) - K.min(y_pred) + K.epsilon())
    return tf.keras.losses.binary_crossentropy(y_true_normalized, y_pred_normalized)

def load_trained_model():
    """Load the trained autoencoder model and optimal threshold."""
    global model, model_file_path, threshold, param, calibration_factor

    # Try different paths for the model
    model_paths = [
        os.path.join(project_root, "model/FNN/model_overall.keras")
    ]
    
    # Find a valid model file
    found_model = False
    for path in model_paths:
        if os.path.exists(path):
            model_file_path = path
            logger.info(f"Found model file: {model_file_path}")
            found_model = True
            break
    
    if not found_model:
        logger.warning("No model file found. Will create a model from scratch.")
        try:
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(model_paths[0]), exist_ok=True)
            
            # Create a new model from the architecture
            input_dim = param["feature"]["n_mels"] * param["feature"]["frames"]
            model = create_model(input_dim, param.get("model", {}).get("architecture", {}))
            
            # Save the model to the expected path
            model_file_path = model_paths[0]
            model.save(model_file_path)
            logger.info(f"Created and saved new model to {model_file_path}")
            
            # Set default threshold
            threshold = 0.35
            logger.info(f"Using default threshold: {threshold}")
            
            # Set calibration factor
            calibration_factor = 0.9
            logger.info(f"Using default calibration factor: {calibration_factor}")
            
            # Test the model with random input
            input_dim = param["feature"]["n_mels"] * param["feature"]["frames"]
            test_input = np.random.random((1, input_dim))
            test_output = model.predict(test_input, verbose=0)
            logger.info(f"Model test successful. Output shape: {test_output.shape}")
            
            return True
        except Exception as e:
            logger.critical(f"Failed to create model: {e}", exc_info=True)
            return False
            
    # Load the model - Try different approaches for compatibility
    try:
        # First make sure TensorFlow is properly imported
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        logger.info(f"Loading pre-trained model directly from: {model_file_path}")
        
        # Try to load the model with custom_objects dictionary
        model = load_model(
            model_file_path, 
            compile=False,
            custom_objects={
                'binary_cross_entropy_loss': binary_cross_entropy_loss
            }
        )
            
        logger.info("Successfully loaded pre-trained model")
        
        # If model loading worked, extract features
        input_dim = param["feature"]["n_mels"] * param["feature"]["frames"]
        logger.info(f"Model input dimension: {input_dim}")
        
        # Compile the model just to ensure it's ready for prediction
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Set threshold to consistent value from the config
        threshold = 0.35
        logger.info(f"Using fixed threshold: {threshold}")
        
        # Set calibration factor - balancing false positives vs false negatives
        calibration_factor = 0.9
        logger.info(f"Using fixed calibration factor: {calibration_factor}")
        
        # Verify model can make predictions
        test_input = np.random.random((1, input_dim))
        test_output = model.predict(test_input, verbose=0)
        logger.info(f"Model test successful. Output shape: {test_output.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model from {model_file_path}: {e}", exc_info=True)
        
        # Try fallback method as last resort
        try:
            logger.info("Trying alternative model loading approach...")
            
            # Load the model with tf.keras.models.load_model directly
            model = tf.keras.models.load_model(model_file_path)
            logger.info("Fallback model loading successful")
            
            # Set threshold and calibration factor
            threshold = 0.35
            calibration_factor = 0.9
            
            return True
        except Exception as e2:
            logger.error(f"Fallback loading also failed: {e2}", exc_info=True)
            
            # If all loading attempts fail, create a new model
            try:
                logger.info("Creating a new model since loading failed")
                input_dim = param["feature"]["n_mels"] * param["feature"]["frames"]
                model = create_model(input_dim, param.get("model", {}).get("architecture", {}))
                
                # Save the new model
                model.save(model_file_path)
                logger.info(f"Created and saved new model to {model_file_path}")
                
                # Set default threshold and calibration factor
                threshold = 0.35
                calibration_factor = 0.9
                
                # Test the model
                test_input = np.random.random((1, input_dim))
                test_output = model.predict(test_input, verbose=0)
                logger.info(f"New model test successful. Output shape: {test_output.shape}")
                
                return True
            except Exception as e3:
                logger.critical(f"Failed to create new model: {e3}", exc_info=True)
                return False

def file_to_vector_array(file_name, n_mels=48, frames=4, n_fft=1024, hop_length=1024, power=2.0):
    """
    Convert file_name (wav file) to a vector array for feature extraction.
    Simplified from the original for use in the API.
    
    Args:
        file_name: Path to the WAV file
        n_mels: Number of mel-banks for feature extraction
        frames: Number of frames to concatenate for every vector element
        n_fft: FFT size for feature extraction
        hop_length: Hop size for feature extraction
        power: Power of the melspectrogram
        
    Returns:
        Vector array in numpy format or empty array if the feature extraction fails
    """
    dims = n_mels * frames
    
    # Load audio file
    try:
        y, sr = librosa.load(file_name, sr=None, mono=True)
        
        # Skip files that are too short
        if len(y) < n_fft:
            logger.warning(f"File too short: {file_name}")
            return np.empty((0, dims), float)
            
        # Extract mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                       sr=sr,
                                                       n_fft=n_fft,
                                                       hop_length=hop_length,
                                                       n_mels=n_mels,
                                                       power=power)
                                                       
        # Convert to log scale (dB)
        log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
        
        # Calculate vector array size
        vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1
        if vectorarray_size < 1:
            logger.warning(f"Not enough frames in {file_name}")
            return np.empty((0, dims), float)

        # Create vector array
        vectorarray = np.zeros((vectorarray_size, dims), float)
        for t in range(frames):
            vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

        # Normalize for binary cross-entropy
        if np.max(vectorarray) > np.min(vectorarray):
            vectorarray = (vectorarray - np.min(vectorarray)) / (np.max(vectorarray) - np.min(vectorarray) + sys.float_info.epsilon)
        
        logger.debug(f"Extracted features from {file_name}: shape={vectorarray.shape}")
        return vectorarray
        
    except Exception as e:
        logger.error(f"Error in file_to_vector_array for {file_name}: {e}")
        return np.empty((0, dims), float)


def analyze_wav(wav_file_path):
    """
    Analyze a WAV file for anomalies.
    
    Args:
        wav_file_path: Path to the WAV file to analyze
        
    Returns:
        Dictionary with analysis results
    """
    global model, threshold, calibration_factor
    
    if model is None:
        logger.error("Model not loaded, cannot analyze file")
        return {"error": "Model not loaded. Please try again later."}
        
    # Get expected label from filename, if it contains "normal" or "abnormal"
    expected_label = None
    filename = os.path.basename(wav_file_path).lower()
    if "abnormal" in filename:
        expected_label = "abnormal"
    elif "normal" in filename:
        expected_label = "normal"
        
    if expected_label:
        logger.info(f"Expected label based on filename: {expected_label}")
    
    try:
        # Extract features from the WAV file
        features = file_to_vector_array(
            wav_file_path,
            n_mels=param["feature"]["n_mels"],
            frames=param["feature"]["frames"],
            n_fft=param["feature"]["n_fft"],
            hop_length=param["feature"]["hop_length"],
            power=param["feature"]["power"]
        )
        
        # If there are no valid features, return an error
        if features.shape[0] == 0:
            logger.warning(f"Could not extract valid features from {wav_file_path}")
            return {"error": "Could not extract valid features from the uploaded file. The file may be too short or corrupted."}
        
        # Log feature statistics
        logger.info(f"Features extracted: shape={features.shape}, min={features.min():.6f}, max={features.max():.6f}")
        
        # Get model predictions (reconstructed features)
        reconstructed = model.predict(features, verbose=0)
        if isinstance(reconstructed, list):
            reconstructed = reconstructed[0]  # Handle different model output formats
        
        # Log reconstruction statistics    
        logger.info(f"Reconstructed: shape={reconstructed.shape}, min={reconstructed.min():.6f}, max={reconstructed.max():.6f}")
        
        # Calculate frame-by-frame errors (MSE)
        frame_errors = np.mean(np.square(features - reconstructed), axis=1)
        logger.info(f"Frame errors: shape={frame_errors.shape}, min={frame_errors.min():.6f}, max={frame_errors.max():.6f}, mean={frame_errors.mean():.6f}")
        
        # Calculate overall anomaly score (mean error across all frames)
        raw_anomaly_score = np.mean(frame_errors)
        logger.info(f"Raw anomaly score: {raw_anomaly_score:.6f}")
        
        # Log some example frame errors for debugging
        logger.info(f"First 5 frame errors: {frame_errors[:5]}")
                
        # Calculate detailed feature difference statistics
        feature_diffs = features - reconstructed
        logger.info(f"Feature diff stats - min: {feature_diffs.min():.6f}, max: {feature_diffs.max():.6f}, mean: {feature_diffs.mean():.6f}")
        
        # Apply calibration factor to get calibrated score
        # Lower calibration factor makes the system more sensitive to anomalies
        # Higher calibration factor makes the system less sensitive
        calibrated_score = raw_anomaly_score * calibration_factor
        logger.info(f"Calibrated score ({calibration_factor}): {calibrated_score:.6f}")
        
        # Determine if this is an anomaly based on threshold
        is_anomaly = calibrated_score >= threshold
        logger.info(f"Is score >= threshold? {calibrated_score:.6f} >= {threshold:.6f} = {is_anomaly}")
        
        # Prediction based on threshold
        prediction = "abnormal" if is_anomaly else "normal"
        logger.info(f"Prediction: {prediction}")
        
        # Log frame-by-frame analysis
        logger.info(f"=== Frame-by-Frame Analysis (first 5 frames) ===")
        for i in range(min(5, len(frame_errors))):
            frame_calibrated = frame_errors[i] * calibration_factor
            frame_anomaly = frame_calibrated >= threshold
            logger.info(f"Frame {i}: Error={frame_errors[i]:.6f}, Calibrated={frame_calibrated:.6f}, Abnormal? {frame_anomaly}")
        
        # Check if prediction is correct, if we know the expected label
        if expected_label:
            is_correct = prediction == expected_label
            logger.info(f"Prediction correct? {is_correct}")
        
        logger.info(f"=== ANALYSIS COMPLETE: {wav_file_path} ===")
        
        # Return the results
        result = {
            "filename": os.path.basename(wav_file_path),
            "prediction": prediction,
            "anomaly_score": float(raw_anomaly_score),
            "calibrated_score": float(calibrated_score),
            "threshold": float(threshold),
            "details": {
                "num_frames": int(features.shape[0]),
                "feature_dims": int(features.shape[1]),
                "min_frame_error": float(frame_errors.min()),
                "max_frame_error": float(frame_errors.max()),
                "frame_error_std": float(np.std(frame_errors))
            }
        }
        
        # If we know the expected label, add it to the results
        if expected_label:
            result["expected_label"] = expected_label
            result["is_correct"] = is_correct
            
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing WAV file {wav_file_path}: {e}", exc_info=True)
        return {"error": f"Error analyzing WAV file: {str(e)}"}

@app.on_event("startup")
async def startup_event():
    """FastAPI startup event: Load the model."""
    logger.info("Application startup: Loading model...")
    
    # Check if model exists first
    for path in [
        os.path.join(project_root, "model/FNN/model_overall.keras"),
        os.path.join(project_root, "model/FNN/model_overall.h5"),
        os.path.join(project_root, "model/FNN/model.h5")
    ]:
        if os.path.exists(path):
            logger.info(f"Found model file at {path}")
            break
    else:
        logger.warning("No model file found! API will create a new model if necessary.")
    
    # Try to load the model
    success = load_trained_model()
    if not success:
        logger.critical("Failed to load or create the model during startup. API will respond with 503 errors.")
    else:
        logger.info("Model loaded successfully. API is ready.")
        
        # Test the model with random data
        try:
            input_dim = param["feature"]["n_mels"] * param["feature"]["frames"]
            test_input = np.random.random((1, input_dim))
            test_output = model.predict(test_input, verbose=0)
            logger.info(f"Model test prediction successful: {test_output.shape}")
        except Exception as e:
            logger.error(f"Model test prediction failed: {e}", exc_info=True)


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
         raise HTTPException(status_code=503, detail="Model is not loaded or failed to load.")

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
        elif "Model not loaded" in analysis_result['error']:
             raise HTTPException(status_code=503, detail="Model is not loaded or failed to load.")
        else:
             raise HTTPException(status_code=500, detail=analysis_result['error'])

    logger.info(f"Prediction successful for {file.filename}")
    
    # Return the full enhanced result with debug information
    return analysis_result


# --- Main Execution ---

if __name__ == '__main__':
    logger.info("Starting FastAPI server...")
    # The model is loaded via the startup_event, no need to call load_trained_model here directly.

    # Get host and port from environment variables or defaults
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", 8000))

    logger.info(f"Server will run on {api_host}:{api_port}")

    # Start the FastAPI app with uvicorn
    uvicorn.run(
        app,  # Use the FastAPI app instance directly
        host=api_host,
        port=api_port,
        log_level="info" # Control uvicorn's logging level
    )

