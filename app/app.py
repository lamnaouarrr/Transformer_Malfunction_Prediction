import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
import requests
import os
import sys
import tempfile
import io
import librosa.display  # For spectrogram display

def file_to_vector_array(file_name, n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2.0):
    """
    Convert file_name to a vector array.
    """
    try:
        # For uploaded files, we need to handle them differently
        if hasattr(file_name, 'read'):
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(file_name.getvalue())
                temp_filename = tmp_file.name
            
            # Load the temporary file
            y, sr = librosa.load(temp_filename, sr=None, mono=True)
            
            # Clean up the temporary file
            os.unlink(temp_filename)
        else:
            # Regular file path
            y, sr = librosa.load(file_name, sr=None, mono=True)
        
        # Calculate mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power
        )
        
        # Convert to log mel spectrogram
        log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + np.finfo(float).eps)
        
        # Create feature vectors by concatenating frames
        vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1
        if vectorarray_size < 1:
            st.error(f"Audio file too short for analysis with current settings. Need at least {frames} frames.")
            return np.empty((0, n_mels * frames), float), None, None
            
        vectorarray = np.zeros((vectorarray_size, n_mels * frames), float)
        for t in range(frames):
            vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T
        
        # Normalize the vector array
        vectorarray = (vectorarray - np.min(vectorarray)) / (np.max(vectorarray) - np.min(vectorarray) + np.finfo(float).eps)
        
        return vectorarray, log_mel_spectrogram, sr
        
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return np.empty((0, n_mels * frames), float), None, None

def load_model_from_path(model_path):
    """Load the Keras model with error handling"""
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def send_to_api(file_content, api_url, mode="file"):
    """Send the file or feature data to the FastAPI endpoint"""
    try:
        if mode == "file":
            # Prepare the file for upload
            files = {'file': ('audio.wav', file_content, 'audio/wav')}
            
            # Send the request
            response = requests.post(api_url, files=files, timeout=10)
        elif mode == "data":
            # Send the feature data directly
            response = requests.post(api_url, json=file_content, timeout=10)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="Abnormal Sound Detector",
        page_icon="🔊",
        layout="wide"
    )
    
    # Title and Introduction
    st.title("🔊 Abnormal Sound Detector")
    st.markdown("Upload machine audio to detect anomalies using an FNN autoencoder.")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Model selection (you can expand this to allow selecting different models)
        model_path = st.text_input(
            "Model Path", 
            value="model/FNN/model_fan_id_00_0dB.h5",
            help="Path to your trained FNN model (.h5 file)"
        )
        
        # API configuration
        api_mode = st.radio(
            "Processing Mode",
            options=["API", "Local Model"],
            index=0,
            help="Choose whether to use the API or local model for processing"
        )
        
        # API URL settings
        if api_mode == "API":
            api_base_url = st.text_input(
                "API Base URL", 
                value="http://localhost:8000",
                help="Base URL of the FastAPI server"
            )
            
            api_endpoint = st.selectbox(
                "API Endpoint",
                options=["predict", "predict_data"],
                index=0,
                help="Endpoint to use: 'predict' for file upload, 'predict_data' for feature vectors"
            )
            
            api_url = f"{api_base_url}/{api_endpoint}"
            
        # Advanced settings expandable section
        with st.expander("Advanced Settings"):
            n_mels = st.slider("Mel Frequency Bins", 16, 128, 64)
            frames = st.slider("Time Frames", 1, 10, 5)
            
        # Health check for API
        if api_mode == "API" and st.button("Check API Connection"):
            try:
                health_url = f"{api_base_url}/health"
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    st.success(f"API is connected and running: {response.json().get('message', 'OK')}")
                else:
                    st.error(f"API returned error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.results = []
            st.success("History cleared!")
    
    # Load the model if using local mode
    if api_mode == "Local Model":
        model = load_model_from_path(model_path)
    else:
        model = None
    
    # Initialize session state for results history
    if "results" not in st.session_state:
        st.session_state.results = []
    
    # Main content area - split into columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Audio Upload Section
        uploaded_file = st.file_uploader("Upload Audio File (WAV format)", type=["wav"])
        
        # Threshold Slider (only for local model)
        if api_mode == "Local Model":
            threshold = st.slider(
                "Detection Sensitivity Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01,
                help="Lower values increase sensitivity (more likely to detect anomalies)"
            )
        
        if uploaded_file is not None:
            # Audio Playback
            st.audio(uploaded_file, format="audio/wav")
            
            # Process button
            process_btn = st.button("📊 Detect Anomaly", use_container_width=True)
            
            if process_btn:
                # Reset file position to beginning
                uploaded_file.seek(0)
                
                # Different processing depending on mode
                if api_mode == "API":
                    with st.spinner("Processing audio via API..."):
                        # Get feature vectors if needed
                        if api_endpoint == "predict_data":
                            vector_array, _, _ = file_to_vector_array(
                                uploaded_file, n_mels=n_mels, frames=frames
                            )
                            if vector_array.size > 0:
                                # Send feature vectors to API
                                result = send_to_api(
                                    {"data": vector_array.tolist()}, 
                                    api_url, 
                                    mode="data"
                                )
                            else:
                                result = None
                        else:
                            # Reset file position
                            uploaded_file.seek(0)
                            # Send file to API
                            result = send_to_api(uploaded_file.getvalue(), api_url, mode="file")
                        
                        if result and "error" not in result:
                            # Display result
                            prediction = result.get("confidence", result.get("prediction", 0))
                            threshold = result.get("threshold", 0.5)
                            prediction_label = result.get("result", result.get("prediction", "unknown"))
                            
                            # Show result with appropriate styling
                            if prediction_label == "normal":
                                st.success(f"✅ Normal Sound (Confidence: {(1-prediction):.4f})")
                            else:
                                st.error(f"⚠️ Abnormal Sound Detected (Anomaly Score: {prediction:.4f})")
                                
                            st.info(f"API Threshold: {threshold:.4f}")
                            
                            # Update Results History
                            st.session_state.results.append({
                                "Filename": uploaded_file.name,
                                "Result": prediction_label.capitalize(),
                                "Score": prediction,
                                "Threshold": threshold,
                                "Source": f"API ({api_endpoint})",
                                "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                
                # Local model processing
                elif api_mode == "Local Model" and model is not None:
                    with st.spinner("Processing audio locally..."):
                        # Reset file position
                        uploaded_file.seek(0)
                        
                        # Process the audio
                        vector_array, log_mel_spectrogram, sr = file_to_vector_array(
                            uploaded_file, n_mels=n_mels, frames=frames
                        )
                        
                        if vector_array.size > 0:
                            # Make prediction
                            prediction = np.mean(model.predict(vector_array, verbose=0))
                            
                            # Result Display
                            result = "Abnormal Sound Detected" if prediction >= threshold else "Normal Sound"
                            
                            # Use colored boxes based on result
                            if result == "Normal Sound":
                                st.success(f"✅ {result} (Confidence: {(1-prediction):.4f})")
                            else:
                                st.error(f"⚠️ {result} (Anomaly Score: {prediction:.4f})")
                                
                            # Update Results History
                            st.session_state.results.append({
                                "Filename": uploaded_file.name,
                                "Result": result,
                                "Score": prediction,
                                "Threshold": threshold,
                                "Source": "Local Model",
                                "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
    
    with col2:
        # Visualizations and results
        if uploaded_file is not None:
            # Reset file position
            uploaded_file.seek(0)
            
            # Display spectrogram if available
            with st.spinner("Generating spectrogram..."):
                vector_array, log_mel_spectrogram, sr = file_to_vector_array(
                    uploaded_file, n_mels=n_mels, frames=frames
                )
                
                if log_mel_spectrogram is not None:
                    st.subheader("Audio Spectrogram")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    if 'librosa.display' in sys.modules:
                        img = librosa.display.specshow(
                            log_mel_spectrogram, 
                            x_axis='time', 
                            y_axis='mel', 
                            sr=sr,
                            fmax=sr/2,
                            ax=ax
                        )
                    else:
                        img = ax.imshow(log_mel_spectrogram, aspect='auto', origin='lower')
                        ax.set_xlabel('Time Frames')
                        ax.set_ylabel('Mel Frequency Bins')
                    
                    fig.colorbar(img, ax=ax, format='%+2.0f dB')
                    ax.set(title='Mel-frequency spectrogram')
                    st.pyplot(fig)
    
    # Results History
    if st.session_state.results:
        st.markdown("---")
        st.subheader("Results History")
        
        # Convert list to DataFrame for better display
        df = pd.DataFrame(st.session_state.results)
        
        # Style the dataframe based on results
        def highlight_results(val):
            if "Abnormal" in str(val):
                return 'background-color: #ffcccc'
            elif "Normal" in str(val):
                return 'background-color: #ccffcc'
            return ''
        
        styled_df = df.style.applymap(highlight_results, subset=['Result'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Export results option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results CSV",
            csv,
            "sound_detection_results.csv",
            "text/csv",
            key='download-csv'
        )
    
    # About Section
    st.markdown("---")
    st.markdown("""
    **About**: This application uses a Feed-Forward Neural Network (FNN) autoencoder for industrial sound anomaly detection. 
    The model is trained to reconstruct normal sounds and flag sounds with high reconstruction error as anomalies.
    """)
    st.markdown("""
    **API Integration**: The application connects to a FastAPI backend (`anomaly_detection_api.py`) 
    which can process either raw audio files or pre-extracted feature vectors.
    """)

if __name__ == "__main__":
    main()