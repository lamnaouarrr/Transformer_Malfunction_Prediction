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
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import tempfile


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add compatibility layer for TensorFlow
import tensorflow as tf
if not hasattr(tf.keras.losses, 'mean_squared_error'):
    # Define the function if it doesn't exist
    tf.keras.losses.mean_squared_error = tf.keras.losses.MeanSquaredError()

# Now import from baseline_fnn
try:
    from baseline_fnn import (
        setup_logging,
        file_to_vector_array,
        demux_wav,
        keras_model,
        normalize_spectrograms
    )
except ImportError as e:
    print(f"Error importing from baseline_fnn: {e}")
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
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Setup logging
logger = setup_logging()

# Find the correct path to the YAML file
yaml_path = os.path.join(os.path.dirname(__file__), '..', 'baseline_fnn.yaml')
with open(yaml_path, "r") as stream:
    param = yaml.safe_load(stream)

# Global variables
model = None
model_file = None
threshold = 0.5  # Default threshold

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def load_model():
    """Load the trained model"""
    global model, model_file, threshold
    
    ensure_directory_exists(f"{param['model_directory']}/FNN")

    # Get model file information from the first file in the directory
    normal_path = Path(param["base_directory"]) / "normal"
    abnormal_path = Path(param["base_directory"]) / "abnormal"
    
    target_dir = normal_path if normal_path.exists() else abnormal_path
    if not target_dir.exists():
        logger.error("Neither normal nor abnormal directory exists!")
        return False
    
    # Get sample files to determine model parameters
    sample_files = list(Path(target_dir).glob(f"*.{param.get('dataset', {}).get('file_extension', 'wav')}"))
    if not sample_files:
        logger.warning(f"No files found in {target_dir}")
        return False
    
    # Parse a sample filename to get db, machine_type, machine_id
    filename = sample_files[0].name
    parts = filename.split('_')
    
    if len(parts) < 4:
        logger.warning(f"Filename format incorrect: {filename}")
        return False
    
    condition = parts[0]  # normal or abnormal
    db = parts[1]
    machine_type = parts[2]
    machine_id = parts[3].split('-')[0] if '-' in parts[3] else parts[3]
    
    # Define model file path
    model_file = f"{param['model_directory']}/FNN/model_{machine_type}_{machine_id}_{db}.h5"
    
    # Check if model file exists
    if not os.path.exists(model_file):
        logger.error(f"Model file not found: {model_file}")
        return False
    
    # Create model
    model_config = param.get("model", {}).get("architecture", {})
    model = keras_model(
        param["feature"]["n_mels"] * param["feature"]["frames"],
        config=model_config
    )
    
    # Load weights
    try:
        model.load_weights(model_file)
        logger.info(f"Model loaded from: {model_file}")
        
        # Load threshold from results if available
        result_file = param.get("result_file", "result_fnn.yaml")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                results = yaml.safe_load(f)
                # Try to get the threshold from results
                if results and "optimal_threshold" in results:
                    threshold = results["optimal_threshold"]
                    logger.info(f"Using optimal threshold from results: {threshold}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def analyze_wav(wav_file_path):
    """
    Analyze a WAV file and predict if it's normal or abnormal
    
    Args:
        wav_file_path: Path to the WAV file
        
    Returns:
        dict: Prediction results
    """
    try:
        # Extract features from WAV file
        data = file_to_vector_array(
            wav_file_path,
            n_mels=param["feature"]["n_mels"],
            frames=param["feature"]["frames"],
            n_fft=param["feature"]["n_fft"],
            hop_length=param["feature"]["hop_length"],
            power=param["feature"]["power"],
            augment=False  # No augmentation during prediction
        )
        
        if data.shape[0] == 0:
            return {"error": f"No valid features extracted from file: {wav_file_path}"}
        
        # Normalize data if specified
        norm_method = param.get("model", {}).get("normalization_method", "minmax")
        if norm_method != "none":
            data = normalize_spectrograms(data, method=norm_method)
        
        # Get predictions
        pred = model.predict(data, verbose=0)
        
        # Average predictions across all frames
        avg_pred = float(np.mean(pred))
        
        # Apply threshold
        is_abnormal = avg_pred >= threshold
        
        return {
            "prediction": "abnormal" if is_abnormal else "normal",
            "confidence": avg_pred,
            "threshold": threshold
        }
    
    except Exception as e:
        logger.error(f"Error analyzing WAV file: {e}")
        return {"error": str(e)}

# API routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "message": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict if a WAV file contains normal or abnormal sound"""
    # Check if file exists
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check if it's a WAV file
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
    
    try:
        # Save the file temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp:
            temp_path = temp.name
            contents = await file.read()
            with open(temp_path, 'wb') as f:
                f.write(contents)
        
        # Analyze the WAV file
        result = analyze_wav(temp_path)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        # Check if there was an error
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alternative endpoint that accepts raw data instead of files
@app.post("/predict_data")
async def predict_data(data: dict):
    """Endpoint to predict using pre-extracted feature data"""
    try:
        # Extract data from request
        if "data" not in data:
            raise HTTPException(status_code=400, detail="No data provided in request")
            
        # Convert to numpy array
        feature_data = np.array(data["data"], dtype=np.float32)
        
        if feature_data.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Empty feature data provided")
            
        # Normalize data if specified
        norm_method = param.get("model", {}).get("normalization_method", "minmax")
        if norm_method != "none":
            feature_data = normalize_spectrograms(feature_data, method=norm_method)
        
        # Get predictions
        pred = model.predict(feature_data, verbose=0)
        
        # Average predictions across all frames
        avg_pred = float(np.mean(pred))
        
        # Apply threshold
        is_abnormal = avg_pred >= threshold
        
        return {
            "prediction": avg_pred,
            "result": "abnormal" if is_abnormal else "normal",
            "threshold": threshold
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def startup():
    """Startup function to load the model"""
    if not load_model():
        logger.error("Failed to load model. Exiting.")
        sys.exit(1)

if __name__ == '__main__':
    # Load model and start the server
    startup()
    # Start the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)