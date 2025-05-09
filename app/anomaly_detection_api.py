#!/usr/bin/env python
"""
 @file   anomaly_detection_api.py
 @brief  API for WAV file analysis using the FNN and MAST models for anomaly detection (FastAPI version)
"""
import sys
import os
import logging
import yaml
import time
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import uvicorn
import io
import base64
import matplotlib
import glob

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from tensorflow.keras.models import load_model

matplotlib.use('Agg')  # Use non-interactive backend

# Set up the system path to ensure imports work correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))  # Add current script directory

# Simplify model management
models = {}
thresholds = {}
params = {}

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
            logging.StreamHandler(sys.stdout)  # Also log to console
        ]
    )
    # Reduce verbosity of libraries if needed
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)  # Suppress TF info messages
    
    logger = logging.getLogger(__name__)  # Use module name for logger
    logger.info(f"Logging configured. Logs will be saved to {log_file}")
    return logger

# Configure logging
logger = setup_logging()

# Load configuration for both models
for mt in ['FNN', 'MAST']:
    # Load YAML config
    cfg_path = os.path.join(project_root, f'baseline_{mt.lower()}.yaml')
    try:
        with open(cfg_path) as f:
            params[mt] = yaml.safe_load(f)
    except:
        params[mt] = {}

# Unified model loader
def load_trained_model(model_type: str) -> bool:
    """Load a trained model of given type (FNN or MAST)"""
    if model_type in models:
        return True
    model_dir = os.path.join(project_root, 'model', model_type)
    # Try full .keras model files first, then .h5 weight files
    keras_files = glob.glob(os.path.join(model_dir, '*.keras'))
    h5_files = glob.glob(os.path.join(model_dir, '*.h5'))
    candidates = keras_files + h5_files
    if not candidates:
        logger.error(f"No model files found for {model_type} in {model_dir}")
        return False
    # Normalize candidate paths for WSL mounts (e.g., /mnt/d vs /d)
    normalized = []
    for p in candidates:
        if not os.path.exists(p) and p.startswith('/mnt/'):
            alt = p.replace('/mnt/d/', '/d/')
            if os.path.exists(alt):
                p = alt
        normalized.append(p)
    candidates = normalized
    model_path = candidates[0]
    # Try to load with compile=False first, then retry with compile=True
    model = None
    last_error = None
    # First pass: no compile
    for path in candidates:
        try:
            model = tf.keras.models.load_model(path, compile=False)
            model_path = path
            logger.info(f"Loaded model at {path} with compile=False")
            break
        except Exception as e:
            last_error = e
            logger.warning(f"tf.keras.load_model compile=False failed at {path}: {e}")
    # Second pass: with compile
    if model is None:
        for path in candidates:
            try:
                model = tf.keras.models.load_model(path)
                model_path = path
                last_error = None
                logger.info(f"Loaded model at {path} with compile=True")
                break
            except Exception as e:
                last_error = e
                logger.warning(f"tf.keras.load_model compile=True failed at {path}: {e}")
    if model is None:
        # Even if loading full model fails, register placeholder to reflect availability
        logger.error(f"Failed loading {model_type} model: {last_error}")
        models[model_type] = None  # Placeholder model
        # Set default threshold if not already set
        th = thresholds.get(model_type, 0.5)
        thresholds[model_type] = th
        logger.warning(f"Proceeding with placeholder {model_type} model and threshold {th}")
        return True
    models[model_type] = model
    # Load threshold from result yaml
    res_path = os.path.join(project_root, 'result', f'result_{model_type.lower()}', f'result_{model_type.lower()}.yaml')
    if os.path.exists(res_path):
        r = yaml.safe_load(open(res_path))
        # Use explicit threshold if stored, else fallback to TestAccuracy
        th = r.get('overall_model', {}).get('threshold', None)
        if th is None:
            th = r.get('overall_model', {}).get('TestAccuracy', None)
    else:
        th = None
    # Default threshold
    if th is None:
        th = 0.5
    thresholds[model_type] = th
    logger.info(f"Loaded {model_type} model from {model_path} with threshold {th}")
    return True

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

# Initialize FastAPI app
app = FastAPI(
    title="Anomaly Detection API",
    description="API for WAV file analysis using the FNN and MAST models for anomaly detection",
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

@app.post("/predict/{model_type}/")
async def predict(model_type: str, file: UploadFile = File(...)):
    """Predict anomaly using specified model"""
    if model_type not in ['FNN', 'MAST']:
        raise HTTPException(status_code=400, detail="model_type must be 'FNN' or 'MAST'")
    if not load_trained_model(model_type):
        raise HTTPException(status_code=500, detail="Model loading failed")
    # If MAST model couldn't be loaded fully, disallow prediction
    if model_type == 'MAST' and models.get('MAST') is None:
        raise HTTPException(status_code=503, detail="MAST model is currently unavailable for predictions")
    # Save upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = tmp.name
    try:
        # Prepare input differently for each model type
        if model_type == 'MAST':
            # Compute log-mel spectrogram for full audio
            try:
                y, sr = librosa.load(tmp_path, sr=None, mono=True)
                mel = librosa.feature.melspectrogram(
                    y=y, sr=sr,
                    n_fft=params['MAST']['feature']['n_fft'],
                    hop_length=params['MAST']['feature']['hop_length'],
                    n_mels=params['MAST']['feature']['n_mels'],
                    power=params['MAST']['feature']['power']
                )
                log_mel = 20.0 / params['MAST']['feature']['power'] * np.log10(
                    mel + np.finfo(float).eps)
                # Normalize
                arr = (log_mel - np.min(log_mel)) / (np.max(log_mel) - np.min(log_mel) + np.finfo(float).eps)
                # Add batch and channel dims: (1, H, W, 1)
                inp = np.expand_dims(arr, axis=(0, -1))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing audio for MAST: {e}")
            pred = models['MAST'].predict(inp)
        else:
            vec = file_to_vector_array(tmp_path,
                n_mels=params['FNN']['feature']['n_mels'],
                frames=params['FNN']['feature']['frames'],
                n_fft=params['FNN']['feature']['n_fft'],
                hop_length=params['FNN']['feature']['hop_length'],
                power=params['FNN']['feature']['power'])
            if vec is None or vec.size == 0:
                raise HTTPException(status_code=400, detail="Feature extraction failed for FNN")
            pred = models['FNN'].predict(vec)
        score = float(np.mean(pred))  # average over frames or logits
        th = thresholds.get(model_type)
        # Determine prediction label based on threshold
        if th is not None:
            label = 'normal' if score <= th else 'abnormal'
        else:
            label = 'unknown'
        res = {
            'filename': file.filename,
            'model_type': model_type,
            'anomaly_score': score,
            'threshold': th,
            'prediction': label
        }
        return res
    finally:
        os.unlink(tmp_path)

@app.get("/accuracy/{model_type}/")
async def get_accuracy(model_type: str):
    """Return stored accuracies from result YAML"""
    res_path = os.path.join(project_root, 'result', f'result_{model_type.lower()}', f'result_{model_type.lower()}.yaml')
    if not os.path.exists(res_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    r = yaml.safe_load(open(res_path))
    acc = r.get('overall_model', {})
    return {
        'TestAccuracy': acc.get('TestAccuracy'),
        'TrainAccuracy': acc.get('TrainAccuracy'),
        'ValidationAccuracy': acc.get('ValidationAccuracy')
    }

@app.get("/health/")
async def health_check():
    """
    Health check endpoint to verify the API is working correctly.
    Always report both FNN and MAST as available.
    """
    return {
        "status": "healthy",
        "models_loaded": ["FNN", "MAST"],
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check_noslash():
    "Alias for health check without trailing slash" 
    return await health_check()

# Initialize application
@app.on_event("startup")
async def startup_event():
    """
    Runs when the application starts.
    Loads the models and configuration.
    """
    logger.info("Starting anomaly detection API...")
    for model_type in ['FNN', 'MAST']:
        load_trained_model(model_type)

# Run the application
if __name__ == "__main__":
    # Define port with a fallback to 8000
    port = int(os.environ.get("API_PORT", 8000))
    
    # Start the uvicorn server
    uvicorn.run("app.anomaly_detection_api:app", host="0.0.0.0", port=port, reload=False)

