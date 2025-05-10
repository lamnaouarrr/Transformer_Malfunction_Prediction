#!/usr/bin/env python3
import subprocess
import sys
import os
import platform
import time
import argparse

def check_python_version():
    """Check if Python version is sufficient (3.7+)"""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 7):
        print(f"❌ Python {major}.{minor} detected. Python 3.7 or higher is required.")
        return False
    print(f"✓ Python {major}.{minor} detected")
    return True

def ensure_pip():
    """Ensure pip is installed"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✓ pip is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ pip not found. Attempting to install pip...")
        
        try:
            # For Python 3.7+
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"], 
                                  stdout=subprocess.PIPE)
            print("✓ pip has been installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install pip using ensurepip")
            
            try:
                # Alternative method: get-pip.py
                print("⚠ Trying alternative pip installation method...")
                import urllib.request
                
                print("Downloading get-pip.py...")
                urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", "get-pip.py")
                
                print("Installing pip...")
                subprocess.check_call([sys.executable, "get-pip.py"], 
                                     stdout=subprocess.PIPE)
                
                # Clean up
                if os.path.exists("get-pip.py"):
                    os.remove("get-pip.py")
                    
                print("✓ pip has been installed successfully")
                return True
            except Exception as e:
                print(f"❌ Failed to install pip: {e}")
                print("\nPlease install pip manually:")
                print("1. Download get-pip.py from https://bootstrap.pypa.io/get-pip.py")
                print("2. Run: python get-pip.py")
                return False

def check_requirements_file(requirements_file):
    """Check if specified requirements file exists"""
    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file '{requirements_file}' not found.")
        print(f"Please make sure '{requirements_file}' exists in the current directory.")
        return False
    else:
        print(f"✓ Found requirements file: {requirements_file}")
    return True

def install_requirements(requirements_file):
    """Install packages from specified requirements file"""
    print(f"\nInstalling packages from {requirements_file}...")
    start_time = time.time()
    
    try:
        # Install all requirements with pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", requirements_file])
        
        # Try to check for GPU if TensorFlow was installed
        try:
            print("\nChecking for GPU support...")
            subprocess.check_call([
                sys.executable, 
                "-c", 
                "import tensorflow as tf; print('Available GPUs:', tf.config.list_physical_devices('GPU'))"
            ], stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, subprocess.SubprocessError):
            print("⚠ Could not verify TensorFlow GPU support (TensorFlow may not be installed)")
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Installation completed in {elapsed_time:.2f} seconds")
        print("\nYou can now run your project!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have sufficient permissions")
        print("2. Check your internet connection")
        print("3. Try installing problematic packages individually")
        print("4. For TensorFlow GPU issues, verify CUDA and cuDNN are properly installed")
        return False

def main():
    """Main setup process"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Install Python dependencies from a requirements file')
    parser.add_argument('-r', '--requirements', 
                        default='requirements.txt', 
                        help='Path to requirements file (default: requirements.txt)')
    args = parser.parse_args()
    
    requirements_file = args.requirements
    
    print("=" * 60)
    print("Python Environment Setup")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Requirements file: {requirements_file}")
    print("=" * 60)
    
    # Step 1: Check Python version
    if not check_python_version():
        print("\nPlease install Python 3.7 or higher and try again.")
        print("Download from: https://www.python.org/downloads/")
        return False
    
    # Step 2: Make sure pip is available
    if not ensure_pip():
        return False
    
    # Step 3: Check requirements file exists
    if not check_requirements_file(requirements_file):
        return False
    
    # Step 4: Install requirements
    if not install_requirements(requirements_file):
        return False
    
    print("\n✓ Setup completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)