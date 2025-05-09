import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os
import sys
import tempfile
import io
import librosa.display

ABOUT_TEXT = """
**About This App**: This predictive maintenance app analyzes machine audio to detect anomalies and reduce downtime.

**Developer:** [Ayoub Lamnaouar](https://www.linkedin.com/in/ayoublamnaouar/)\n
**Repository:** [GitHub](https://github.com/lamnaouarrr/Transformer_Malfunction_Prediction)\n
**Advisor:** Prof. Jun Zhang (张俊)\n
**Models:** FNN & MAST (~87.6% & ~83% accuracy)\n
**Dataset:** [MIMII](https://arxiv.org/abs/1909.09347)\n
**Built with:** TensorFlow 2.x, FastAPI, Streamlit
"""
st.set_page_config(
    page_title="Abnormal Sound Detector",
    page_icon="🔊",
    layout="wide",
    menu_items={"About": ABOUT_TEXT}
)

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

def send_to_api(file_content, api_url, mode="file"):
    """Send the file or feature data to the FastAPI endpoint"""
    try:
        if mode == "file":
            # Prepare the file for upload
            files = {'file': ('audio.wav', file_content, 'audio/wav')}
            
            # Send the request with a longer timeout (30 seconds instead of 10)
            response = requests.post(api_url, files=files, timeout=30)
        elif mode == "data":
            # Send the feature data directly with a longer timeout
            response = requests.post(api_url, json=file_content, timeout=30)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("API request timed out. The server might be processing a large file or under heavy load.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API server. Please check if the server is running at the correct address.")
        return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None

def main():
    # Title and Introduction
    st.title("🔊 Abnormal Sound Detector")
    st.markdown("Upload machine audio to detect anomalies using an API-based detector.")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # API configuration
        api_base_url = st.text_input(
            "API Base URL", 
            value="http://localhost:8000",
            help="Base URL of the FastAPI server"
        )
        
        # Select model type
        model_type = st.selectbox(
            "Model Type",
            options=["FNN", "MAST"],
            index=0,
            help="Choose which trained model to use"
        )
        # Build prediction URL
        api_url = f"{api_base_url}/predict/{model_type}/"
        
        # Spectrogram settings expandable section
        with st.expander("Spectrogram Settings"):
            n_mels = st.slider("Mel Frequency Bins", 16, 128, 64)
            frames = st.slider("Sliding Window Frames", 1, 10, 5)
            hop_length = st.slider("Hop Length", 64, 2048, 512, step=64)
        # Get model accuracy
        if st.button("Get Model Accuracy"):
            try:
                acc = requests.get(f"{api_base_url}/accuracy/{model_type}/").json()
                st.success(f"Test: {acc['TestAccuracy']:.4f}, Train: {acc['TrainAccuracy']:.4f}, Val: {acc['ValidationAccuracy']:.4f}")
            except Exception as e:
                st.error(f"Failed to get accuracy: {e}")
        
        # Health check for API
        if st.button("Check API Connection"):
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
    
    # Initialize session state for results history
    if "results" not in st.session_state:
        st.session_state.results = []
    
    # Main content area - split into columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Audio Upload Section
        uploaded_file = st.file_uploader("Upload Audio File (WAV format)", type=["wav"])
        
        if uploaded_file is not None:
            # Audio Playback
            st.audio(uploaded_file, format="audio/wav")
            
            # Process button (uses selected model)
            process_btn = st.button("📊 Detect Anomaly", use_container_width=True)
            
            if process_btn:
                # Reset file position to beginning
                uploaded_file.seek(0)
                with st.spinner(f"Processing audio via API ({model_type})..."):
                    # Send file to API
                    result = send_to_api(uploaded_file.getvalue(), api_url, mode="file")
                
                if result and "error" not in result:
                    # Get the anomaly score and prediction label from the API result
                    anomaly_score = result.get("anomaly_score", 0.0) # Use get with a default float value
                    prediction_label = result.get("prediction", "unknown") # Get the string label
                    threshold = result.get("threshold", 0.5) # Get the threshold from the API

                    # Show result with appropriate styling
                    if prediction_label == "normal":
                        st.success(f"✅ Normal Sound (Anomaly Score: {anomaly_score:.4f})")
                    else:
                        st.error(f"⚠️ Abnormal Sound Detected (Anomaly Score: {anomaly_score:.4f})")

                    st.info(f"API Threshold: {threshold:.4f}")

                    # Update Results History
                    st.session_state.results.append({
                        "Filename": uploaded_file.name,
                        "Result": prediction_label.capitalize(),
                        "Score": anomaly_score, # Use the numerical score for history
                        "Threshold": threshold,
                        "Source": f"API ({model_type})",
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
                    uploaded_file, n_mels=n_mels, frames=frames, hop_length=hop_length
                )
                
                if log_mel_spectrogram is not None:
                    st.subheader("Audio Spectrogram")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    if 'librosa.display' in sys.modules:
                        img = librosa.display.specshow(
                            log_mel_spectrogram,
                            x_axis='time', y_axis='mel', sr=sr,
                            hop_length=hop_length, fmax=sr/2,
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

if __name__ == "__main__":
    main()