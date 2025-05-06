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
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
from typing import Optional, Dict, Any, List
import uvicorn
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

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
threshold = 0.5  # Default threshold value (will be updated from results if available)
calibration_factor = 1.0  # Default calibration factor (no adjustment)

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
        os.path.join(project_root, "model/FNN/model_overall.keras"),
        os.path.join(project_root, "model/FNN/model_overall.h5"),
        os.path.join(project_root, "model/FNN/model.h5"),
        os.path.join(project_root, "model", "FNN", "model_overall.keras")  # Added alternate path format
    ]
    
    # Log all available model files
    logger.info("Searching for model files...")
    model_dir = os.path.join(project_root, "model", "FNN")
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            logger.info(f"Found file in model directory: {file}")
    
    # Find a valid model file
    found_model = False
    for path in model_paths:
        if os.path.exists(path):
            model_file_path = path
            logger.info(f"Found model file: {model_file_path}")
            found_model = True
            break
    
    if not found_model:
        logger.critical("No model file found. API cannot start without an existing model.")
        return False
            
    # Load the model if it exists
    try:
        # First make sure TensorFlow is properly imported
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        logger.info(f"Loading pre-trained model directly from: {model_file_path}")
        
        # Try to load the model with custom_objects dictionary
        try:
            # First try loading with compile=True to preserve original compilation
            model = load_model(
                model_file_path, 
                compile=True,
                custom_objects={
                    'binary_cross_entropy_loss': binary_cross_entropy_loss
                }
            )
            logger.info("Successfully loaded pre-trained model with original compilation")
        except Exception as e:
            logger.warning(f"Could not load model with compile=True: {e}")
            logger.info("Trying to load model without compilation...")
            
            # If that fails, try without compilation
            model = load_model(
                model_file_path, 
                compile=False,
                custom_objects={
                    'binary_cross_entropy_loss': binary_cross_entropy_loss
                }
            )
            logger.info("Successfully loaded pre-trained model without compilation")
            
            # Compile the model with settings from yaml file
            logger.info("Compiling model with settings from yaml...")
            compile_params = param.get("fit", {}).get("compile", {}).copy()
            
            # Handle learning_rate separately for the optimizer
            learning_rate = compile_params.pop("learning_rate", 0.001) if "learning_rate" in compile_params else 0.001
            
            if "optimizer" in compile_params and compile_params["optimizer"] == "adam":
                compile_params["optimizer"] = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
            # Set loss function
            loss_type = param.get("model", {}).get("loss", "binary_crossentropy")
            if loss_type == "binary_crossentropy":
                compile_params["loss"] = binary_cross_entropy_loss
            else:
                compile_params["loss"] = "binary_crossentropy"
            
            # Add metrics
            compile_params["metrics"] = ['accuracy']
            
            # Compile the model
            model.compile(**compile_params)
            logger.info("Model compiled successfully")
        
        # Extract features dimension
        input_dim = param["feature"]["n_mels"] * param["feature"]["frames"]
        logger.info(f"Model input dimension: {input_dim}")
        
        # Check multiple possible locations for the results file
        result_files = [
            os.path.join(project_root, "result/result_fnn/result_fnn.yaml"),
            os.path.join(project_root, "result/result_fnn.yaml")
        ]
        
        # First try to find the optimal threshold in the results file
        found_threshold = False
        for result_file in result_files:
            if os.path.exists(result_file):
                try:
                    with open(result_file, "r") as f:
                        results = yaml.safe_load(f)
                    
                    logger.info(f"Results file found at: {result_file}")
                    logger.info(f"Results keys: {list(results.keys())}")
                    
                    # First try to find the optimal_threshold directly
                    if "overall_model" in results:
                        logger.info(f"Found 'overall_model' key in results")
                        if "optimal_threshold" in results["overall_model"]:
                            threshold = float(results["overall_model"]["optimal_threshold"])
                            found_threshold = True
                            logger.info(f"Using optimal threshold from results: {threshold}")
                    
                    # If we didn't find it, look in other locations
                    if not found_threshold:
                        # Try direct top-level key
                        if "optimal_threshold" in results:
                            threshold = float(results["optimal_threshold"])
                            found_threshold = True
                            logger.info(f"Using optimal threshold from top level: {threshold}")
                        else:
                            # Look through all machine keys for thresholds and use the average
                            thresholds = []
                            for key, value in results.items():
                                if isinstance(value, dict) and "optimal_threshold" in value:
                                    thresholds.append(float(value["optimal_threshold"]))
                            
                            if thresholds:
                                threshold = sum(thresholds) / len(thresholds)
                                found_threshold = True
                                logger.info(f"Using average of {len(thresholds)} thresholds: {threshold}")
                    
                    if found_threshold:
                        break  # Stop looking at other files if we found a threshold
                    
                except Exception as e:
                    logger.warning(f"Error reading results file {result_file}: {e}")
        
        # If we still haven't found a threshold, use the default model threshold configuration
        if not found_threshold:
            if param.get("model", {}).get("threshold", None) is not None:
                threshold = float(param["model"]["threshold"])
                logger.info(f"Using threshold from configuration: {threshold}")
            else:
                logger.info(f"No optimal threshold found, using default: {threshold}")
        
        # Log the model summary
        model.summary(print_fn=lambda x: logger.info(x))
        logger.info(f"Model loaded successfully with threshold: {threshold}")
        return True
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.exception("Detailed stack trace:")
        return False

def file_to_vector_array(file_path,
                        n_mels=64,
                        frames=5,
                        n_fft=1024,
                        hop_length=512,
                        power=2.0):
    """
    Convert audio file to vector array for model input.
    """
    logger.debug(f"Processing file: {file_path}")
    # Audio file loading
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
        if y.ndim > 1:  # If multi-channel, use the first channel
            y = y[0, :]
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        return None
    
    # Feature extraction
    try:
        # Calculate mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                       sr=sr,
                                                       n_fft=n_fft,
                                                       hop_length=hop_length,
                                                       n_mels=n_mels,
                                                       power=power)
        
        # Convert to log scale
        log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
        
        # Calculate the number of frames that can be included
        vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1
        if vectorarray_size < 1:
            logger.warning(f"Audio file {file_path} is too short for feature extraction!")
            return None
        
        # Create the feature vectors
        dims = n_mels * frames
        vectorarray = np.zeros((vectorarray_size, dims), float)
        for t in range(frames):
            vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T
        
        # Normalize the vectors (same as in training)
        if np.max(vectorarray) > np.min(vectorarray):
            vectorarray = (vectorarray - np.min(vectorarray)) / (np.max(vectorarray) - np.min(vectorarray) + sys.float_info.epsilon)
            
        # Log successful feature extraction
        logger.debug(f"Extracted features from {file_path}: shape={vectorarray.shape}")
        return vectorarray
    
    except Exception as e:
        logger.error(f"Error in feature extraction for {file_path}: {e}")
        logger.exception("Detailed stack trace:")
        return None

def predict_anomaly(file_path):
    """
    Predict if the audio file contains an anomaly.
    
    Args:
        file_path: Path to the audio file to analyze
        
    Returns:
        Dictionary with prediction results
    """
    global model, threshold, param, calibration_factor
    
    start_time = time.time()
    
    # Check if model is loaded
    if model is None:
        logger.error("Model not loaded. Cannot make prediction.")
        return {"error": "Model not loaded"}
    
    # Extract features from the audio file
    try:
        features = file_to_vector_array(
            file_path,
            n_mels=param["feature"]["n_mels"],
            frames=param["feature"]["frames"],
            n_fft=param["feature"]["n_fft"],
            hop_length=param["feature"]["hop_length"],
            power=param["feature"]["power"]
        )
        
        if features is None or features.shape[0] == 0:
            logger.error(f"Failed to extract features from {file_path}")
            return {
                "error": "Feature extraction failed",
                "is_anomaly": True,  # Conservatively mark as anomaly
                "anomaly_score": 1.0,
                "processing_time": time.time() - start_time
            }
            
        # Make prediction
        predictions = model.predict(features, verbose=0)
        
        # Average the predictions
        anomaly_score = float(np.mean(predictions))
        
        # Apply calibration factor to the threshold
        effective_threshold = threshold * calibration_factor
        
        is_anomaly = anomaly_score >= effective_threshold
        
        # Log prediction details
        logger.info(f"File: {os.path.basename(file_path)}")
        logger.info(f"Anomaly score: {anomaly_score:.6f}")
        logger.info(f"Threshold: {effective_threshold:.6f}")
        logger.info(f"Is anomaly: {is_anomaly}")
        
        processing_time = time.time() - start_time
        
        return {
            "filename": os.path.basename(file_path),
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(anomaly_score),
            "threshold": float(effective_threshold),
            "processing_time": float(processing_time)
        }
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        logger.exception("Detailed stack trace:")
        return {
            "error": f"Prediction error: {str(e)}",
            "is_anomaly": True,  # Conservatively mark as anomaly
            "anomaly_score": 1.0,
            "processing_time": time.time() - start_time
        }

# API routes
@app.get("/")
async def root():
    """Root endpoint that returns API information"""
    return {
        "name": "Anomaly Detection API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model is not None,
        "model_file": model_file_path if model is not None else None,
        "threshold": threshold
    }

@app.post("/predict/")
async def predict_from_upload(file: UploadFile = File(...)):
    """
    Endpoint to receive an uploaded WAV file and predict if it contains an anomaly.
    """
    # Check if model is loaded
    if model is None:
        if not load_trained_model():
            raise HTTPException(status_code=500, detail="Model loading failed")
    
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        # Save uploaded file to temp file
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Process the file
        result = predict_anomaly(temp_path)
        # Add the original filename to the result
        result["original_filename"] = file.filename
        return result
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/predict_local/")
async def predict_from_local_path(request: Request):
    """
    Endpoint to process a local file specified by its path.
    
    This is primarily for internal use when the file is already on the server.
    """
    # Check if model is loaded
    if model is None:
        if not load_trained_model():
            raise HTTPException(status_code=500, detail="Model loading failed")
    
    # Parse the request body
    data = await request.json()
    file_path = data.get("file_path")
    
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path is required")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    # Process the file
    result = predict_anomaly(file_path)
    return result

@app.post("/set_threshold/")
async def set_threshold(request: Request):
    """
    Endpoint to update the anomaly threshold.
    """
    global threshold
    
    # Parse the request body
    data = await request.json()
    new_threshold = data.get("threshold")
    
    if new_threshold is None:
        raise HTTPException(status_code=400, detail="threshold parameter is required")
    
    try:
        new_threshold = float(new_threshold)
        if new_threshold < 0 or new_threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        # Update the threshold
        old_threshold = threshold
        threshold = new_threshold
        
        logger.info(f"Threshold updated from {old_threshold} to {new_threshold}")
        
        return {
            "status": "success",
            "old_threshold": old_threshold,
            "new_threshold": new_threshold
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/set_calibration/")
async def set_calibration(request: Request):
    """
    Endpoint to update the calibration factor for the threshold.
    """
    global calibration_factor
    
    # Parse the request body
    data = await request.json()
    new_factor = data.get("factor")
    
    if new_factor is None:
        raise HTTPException(status_code=400, detail="factor parameter is required")
    
    try:
        new_factor = float(new_factor)
        if new_factor <= 0:
            raise ValueError("Calibration factor must be positive")
        
        # Update the calibration factor
        old_factor = calibration_factor
        calibration_factor = new_factor
        
        logger.info(f"Calibration factor updated from {old_factor} to {new_factor}")
        
        return {
            "status": "success",
            "old_factor": old_factor,
            "new_factor": new_factor,
            "effective_threshold": threshold * calibration_factor
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health/")
async def health_check():
    """
    Health check endpoint to verify the API is working correctly.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

@app.get("/debug/")
async def debug_info():
    """
    Debug endpoint to return information about the model and configuration.
    Only for development use.
    """
    return {
        "model_file": model_file_path,
        "model_loaded": model is not None,
        "threshold": threshold,
        "calibration_factor": calibration_factor,
        "effective_threshold": threshold * calibration_factor,
        "feature_config": param.get("feature", {}),
        "tensorflow_version": tf.__version__,
        "project_root": project_root
    }

@app.post("/generate_spectrogram/")
async def generate_spectrogram(file: UploadFile = File(...)):
    """
    Generate and return a spectrogram image from an uploaded WAV file.
    
    Args:
        file: The uploaded WAV file
        
    Returns:
        Spectrogram image as PNG
    """
    logger.info(f"Generating spectrogram for file: {file.filename}")
    
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        # Save uploaded file to temp file
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Load the audio file
        y, sr = librosa.load(temp_path, sr=None)
        
        # Create figure and axes
        plt.figure(figsize=(10, 6))
        
        # Generate mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_mels=param["feature"]["n_mels"],
            n_fft=param["feature"]["n_fft"],
            hop_length=param["feature"]["hop_length"],
            power=param["feature"]["power"]
        )
        
        # Convert to log scale (dB)
        log_S = librosa.power_to_db(S, ref=np.max)
        
        # Plot spectrogram
        img = librosa.display.specshow(
            log_S, 
            sr=sr, 
            x_axis='time', 
            y_axis='mel', 
            hop_length=param["feature"]["hop_length"],
            cmap='viridis'
        )
        
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel spectrogram - {file.filename}')
        plt.tight_layout()
        
        # Save to BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Return the image
        return StreamingResponse(buf, media_type="image/png")
    
    except Exception as e:
        logger.error(f"Error generating spectrogram: {e}")
        logger.exception("Detailed stack trace:")
        raise HTTPException(status_code=500, detail=f"Error generating spectrogram: {str(e)}")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/predict_with_spectrogram/")
async def predict_with_spectrogram(file: UploadFile = File(...)):
    """
    Predict if the audio file contains an anomaly and return the spectrogram.
    
    Args:
        file: The uploaded WAV file
        
    Returns:
        JSON response with prediction results and base64 encoded spectrogram image
    """
    # Check if model is loaded
    if model is None:
        if not load_trained_model():
            raise HTTPException(status_code=500, detail="Model loading failed")
    
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        # Save uploaded file to temp file
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Process the file for prediction
        prediction_result = predict_anomaly(temp_path)
        
        # Add the original filename to the result
        prediction_result["original_filename"] = file.filename
        
        # Generate spectrogram
        try:
            # Load the audio file
            y, sr = librosa.load(temp_path, sr=None)
            
            # Create figure and axes
            plt.figure(figsize=(10, 6))
            
            # Generate mel spectrogram
            S = librosa.feature.melspectrogram(
                y=y, 
                sr=sr,
                n_mels=param["feature"]["n_mels"],
                n_fft=param["feature"]["n_fft"],
                hop_length=param["feature"]["hop_length"],
                power=param["feature"]["power"]
            )
            
            # Convert to log scale (dB)
            log_S = librosa.power_to_db(S, ref=np.max)
            
            # Plot spectrogram
            img = librosa.display.specshow(
                log_S, 
                sr=sr, 
                x_axis='time', 
                y_axis='mel', 
                hop_length=param["feature"]["hop_length"],
                cmap='viridis'
            )
            
            plt.colorbar(format='%+2.0f dB')
            title = f'Mel spectrogram - {file.filename}'
            if prediction_result.get("is_anomaly", False):
                title += " - ANOMALY DETECTED"
            plt.title(title)
            plt.tight_layout()
            
            # Save to BytesIO
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plt.close()
            
            # Encode the image as base64
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            # Add the image to the result
            prediction_result["spectrogram_base64"] = img_str
            
        except Exception as e:
            logger.error(f"Error generating spectrogram: {e}")
            prediction_result["spectrogram_error"] = str(e)
        
        return prediction_result
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# Initialize application
@app.on_event("startup")
async def startup_event():
    """
    Runs when the application starts.
    Loads the model and configuration.
    """
    logger.info("Starting anomaly detection API...")
    
    # Load the trained model
    if not load_trained_model():
        logger.warning("Starting without a model. Model will need to be loaded later.")
    else:
        logger.info("Model loaded successfully during startup.")

# Run the application
if __name__ == "__main__":
    # Define port with a fallback to 8000
    port = int(os.environ.get("API_PORT", 8000))
    
    # Start the uvicorn server
    uvicorn.run("anomaly_detection_api:app", host="0.0.0.0", port=port, reload=False)

