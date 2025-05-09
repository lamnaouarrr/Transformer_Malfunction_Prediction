import subprocess
import sys
import time
import platform

def install_requirements():
    print("=" * 50)
    print("Installing dependencies for deep learning with audio processing...")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version}")
    print("=" * 50)
    
    try:
        # First check for CUDA availability
        subprocess.run(["nvidia-smi"], check=True, capture_output=True)
        print("✓ NVIDIA GPU detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ No NVIDIA GPU detected or nvidia-smi not available")
        print("  Installation will continue but GPU acceleration may not be available")
    
    print("\nInstalling packages from requirements.txt...")
    start_time = time.time()
    
    try:
        # Install all requirements with pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"])
        
        # Verify TensorFlow can see the GPU
        print("\nVerifying TensorFlow GPU access...")
        subprocess.check_call([
            sys.executable, 
            "-c", 
            "import tensorflow as tf; print('Available GPUs:', tf.config.list_physical_devices('GPU'))"
        ])
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Installation completed successfully in {elapsed_time:.2f} seconds")
        print("\nYou can now run your deep learning scripts!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Installation failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have sufficient permissions")
        print("2. Check your internet connection")
        print("3. Try installing problematic packages individually")
        print("4. For TensorFlow GPU issues, verify CUDA and cuDNN are properly installed")
        return False
    
    return True

if __name__ == "__main__":
    success = install_requirements()
    sys.exit(0 if success else 1)