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
import uvicorn
import glob

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

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
threshold = 0.35  # Base threshold value
calibration_factor = 0.5  # Reduced calibration factor - to reduce false positives

# Import functions from baseline_fnn.py
try:
    from baseline_fnn import file_to_vector_array, keras_model
    print("✓ Successfully imported core functions from baseline_fnn.py")
except ImportError as e:
    print(f"Warning: Could not import all functions from baseline_fnn.py: {e}")
    print("Falling back to local function implementations")
    
    # Define minimal versions of the required functions locally
    def file_to_vector_array(file_name, 
                           n_mels=64,
                           frames=5,
                           n_fft=1024,
                           hop_length=512,
                           power=2.0,
                           augment=False,
                           param=None):
        """
        Convert file_name to a vector array with EXACT SAME normalization as used in training.
        This function is replicated from baseline_fnn.py to ensure consistency.
        """
        dims = n_mels * frames
        
        try:
            # Use librosa directly to ensure consistent loading
            y, sr = librosa.load(file_name, sr=param["feature"]["sr"], mono=True)
            
            if y is None or len(y) == 0:
                print(f"Error loading {file_name}")
                return np.empty((0, dims), float)
            
            # Skip files that are too short
            if len(y) < n_fft:
                print(f"File too short: {file_name}")
                return np.empty((0, dims), float)
                
            # Extract mel-spectrogram using EXACT SAME parameters from config
            mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                           sr=sr,
                                                           n_fft=n_fft,
                                                           hop_length=hop_length,
                                                           n_mels=n_mels,
                                                           power=power)
            
            # Use the EXACT SAME log transformation as in baseline_fnn.py
            log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
            
            vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1
            if vectorarray_size < 1:
                print(f"Not enough frames in {file_name}")
                return np.empty((0, dims), float)

            # Use a more memory-efficient approach for feature extraction
            vectorarray = np.zeros((vectorarray_size, dims), float)
            for t in range(frames):
                vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T

            # Apply normalization using the method specified in config
            norm_method = param.get("model", {}).get("normalization_method", "minmax_local")
            
            if norm_method == "minmax_local":
                # Local min-max normalization (per file)
                if np.max(vectorarray) > np.min(vectorarray):
                    vectorarray = (vectorarray - np.min(vectorarray)) / (np.max(vectorarray) - np.min(vectorarray) + sys.float_info.epsilon)
            elif norm_method == "minmax_global":
                # Apply global min-max normalization if available
                # This would require global min/max values from training
                print("Warning: Global min-max normalization not implemented, using local normalization")
                if np.max(vectorarray) > np.min(vectorarray):
                    vectorarray = (vectorarray - np.min(vectorarray)) / (np.max(vectorarray) - np.min(vectorarray) + sys.float_info.epsilon)
            elif norm_method == "standard":
                # Standard normalization (zero mean, unit variance)
                if np.std(vectorarray) > 0:
                    vectorarray = (vectorarray - np.mean(vectorarray)) / (np.std(vectorarray) + sys.float_info.epsilon)
            
            # Apply subsampling to reduce memory usage, exactly as in baseline_fnn.py
            if param and "feature" in param and "stride" in param["feature"]:
                stride = param["feature"]["stride"]
                vectorarray = vectorarray[::stride, :]
            
            return vectorarray
            
        except Exception as e:
            print(f"Error in file_to_vector_array for {file_name}: {e}")
            return np.empty((0, dims), float)

    def keras_model(input_dim, config=None):
        """Define a basic keras model when baseline_fnn.py import fails"""
        if config is None:
            config = {}
        
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense
        from tensorflow.keras.regularizers import l2
        
        # Extract parameters from config with defaults
        depth = config.get("depth", 3)
        width = config.get("width", 128)
        bottleneck = config.get("bottleneck", 16)
        activation = config.get("activation", "relu")
        
        inputLayer = Input(shape=(input_dim,))
        x = inputLayer
        
        # Encoder
        for i in range(depth):
            layer_width = max(width // (2 ** i), bottleneck) if i < depth - 1 else bottleneck
            x = Dense(layer_width, activation=activation, kernel_regularizer=l2(1e-5))(x)
        
        # Decoder
        for i in range(depth - 1, -1, -1):
            if i == 0:
                x = Dense(input_dim, activation=None)(x)
            else:
                layer_width = max(width // (2 ** (i - 1)), bottleneck)
                x = Dense(layer_width, activation=activation, kernel_regularizer=l2(1e-5))(x)
        
        model = Model(inputs=inputLayer, outputs=x)
        return model

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
        logger.error("No model file found. Cannot continue.")
        return False
            
    # Load the model - Try different approaches for compatibility
    try:
        # First make sure TensorFlow is properly imported
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        # Define custom loss for loading
        @tf.keras.utils.register_keras_serializable()
        def binary_cross_entropy_loss(y_true, y_pred):
            """Binary cross-entropy loss for autoencoder"""
            import tensorflow.keras.backend as K
            # Normalize inputs if needed (values between 0 and 1)
            y_true_normalized = (y_true - K.min(y_true)) / (K.max(y_true) - K.min(y_true) + K.epsilon())
            y_pred_normalized = (y_pred - K.min(y_pred)) / (K.max(y_pred) - K.min(y_pred) + K.epsilon())
            return tf.keras.losses.binary_crossentropy(y_true_normalized, y_pred_normalized)
        
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
    global model, threshold, param, calibration_factor # Ensure access to globals

    if model is None:
         logger.error("Model is not loaded. Cannot analyze.")
         return {"error": "Model not loaded"}

    try:
        logger.info(f"=== ANALYSIS START: {wav_file_path} ===")
        logger.info(f"Using threshold: {threshold:.6f}, calibration_factor: {calibration_factor:.6f}")

        # Extract features
        features = file_to_vector_array(
            wav_file_path,
            n_mels=param["feature"]["n_mels"],
            frames=param["feature"]["frames"],
            n_fft=param["feature"]["n_fft"],
            hop_length=param["feature"]["hop_length"],
            power=param["feature"]["power"],
            augment=False,  # No augmentation during inference
            param=param     # Pass the parameters
        )

        if features.shape[0] == 0:
            logger.warning(f"No features extracted from {wav_file_path}. Cannot predict.")
            return {"error": f"Could not extract valid features from file: {os.path.basename(wav_file_path)}"}

        logger.info(f"Features extracted: shape={features.shape}, min={features.min():.6f}, max={features.max():.6f}")

        # Get reconstructed features from the model
        reconstructed_features = model.predict(features, verbose=0)
        logger.info(f"Reconstructed: shape={reconstructed_features.shape}, min={reconstructed_features.min():.6f}, max={reconstructed_features.max():.6f}")
        
        # Calculate mean squared error for each frame
        frame_errors = np.mean(np.square(features - reconstructed_features), axis=1)
        logger.info(f"Frame errors: shape={frame_errors.shape}, min={frame_errors.min():.6f}, max={frame_errors.max():.6f}, mean={np.mean(frame_errors):.6f}")
        
        # Calculate raw anomaly score
        raw_anomaly_score = float(np.mean(frame_errors))
        logger.info(f"Raw anomaly score: {raw_anomaly_score:.6f}")
        
        # Log first few frames' errors to check consistency
        logger.info(f"First 5 frame errors: {frame_errors[:5]}")
                
        # Check if the file is supposed to be normal or abnormal based on filename
        filename = os.path.basename(wav_file_path)
        expected_label = "abnormal" if "abnormal" in filename.lower() else "normal"
        logger.info(f"Expected label based on filename: {expected_label}")
        
        # Apply calibration factor
        anomaly_score = raw_anomaly_score * calibration_factor
        logger.info(f"Calibrated score ({calibration_factor}): {anomaly_score:.6f}")
        
        # Debug info about the feature vectors and their differences
        feature_diff = features - reconstructed_features
        logger.info(f"Feature diff stats - min: {np.min(feature_diff):.6f}, max: {np.max(feature_diff):.6f}, mean: {np.mean(feature_diff):.6f}")
        
        # Compare score with the threshold
        is_abnormal = anomaly_score >= threshold
        prediction_label = "abnormal" if is_abnormal else "normal"
        logger.info(f"Is score >= threshold? {anomaly_score:.6f} >= {threshold:.6f} = {is_abnormal}")
        logger.info(f"Prediction: {prediction_label}")
        
        # Print frame-by-frame analysis for the first few frames
        logger.info("=== Frame-by-Frame Analysis (first 5 frames) ===")
        for i in range(min(5, len(frame_errors))):
            logger.info(f"Frame {i}: Error={frame_errors[i]:.6f}, Calibrated={frame_errors[i]*calibration_factor:.6f}, Abnormal? {frame_errors[i]*calibration_factor >= threshold}")
        
        # Check if prediction matches expected label based on filename
        prediction_correct = prediction_label == expected_label
        logger.info(f"Prediction correct? {prediction_correct}")
        logger.info(f"=== ANALYSIS COMPLETE: {wav_file_path} ===")

        return {
            "file_name": os.path.basename(wav_file_path),
            "prediction": prediction_label,
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "raw_score": raw_anomaly_score,
            "expected_label": expected_label,
            "prediction_correct": prediction_correct,
            "num_frames": int(features.shape[0]),
            "frame_errors_stats": {
                "min": float(frame_errors.min()),
                "max": float(frame_errors.max()),
                "mean": float(np.mean(frame_errors)),
                "std": float(np.std(frame_errors))
            }
        }

    except Exception as e:
        logger.error(f"Error analyzing WAV file {wav_file_path}: {e}", exc_info=True)
        return {"error": f"Analysis failed for {os.path.basename(wav_file_path)}: {str(e)}"}

# --- FastAPI Routes ---

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
        logger.critical("No model file found! API cannot function.")
        # We'll continue but API will return errors
    
    # Try to load the model
    success = load_trained_model()
    if not success:
        logger.critical("Failed to load the model during startup. API will respond with 503 errors.")
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

