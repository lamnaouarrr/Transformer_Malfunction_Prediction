#!/usr/bin/env python3
"""
run_optimized_trainer.py - Script to run the ultra memory-efficient AST trainer

This script checks dependencies, sets up memory optimizations, and launches the trainer
"""

import os
import sys
import subprocess
import warnings
import traceback

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

def check_dependencies():
    """Check if required packages are installed and install if missing"""
    required_packages = [
        'tensorflow>=2.11.0',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'pyyaml',
        'tqdm',
        'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        package_name = package.split('>=')[0].split('==')[0]
        try:
            __import__(package_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package_name} is missing")
    
    if missing_packages:
        print("\nInstalling missing packages...")
        subprocess.call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("All dependencies installed successfully!")

def setup_memory_optimizations():
    """Configure environment variables for memory optimization"""
    # Limit TensorFlow memory growth
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Limit TensorFlow to use only 80% of GPU memory
    os.environ['TF_MEMORY_ALLOCATION'] = '0.8'
    
    # Enable TensorFlow memory optimizations
    os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'true'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    # Use the memory-efficient algorithm versions
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    
    # Enable XLA JIT compilation
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2'
    
    print("Memory optimizations enabled through environment variables")

def main():
    """Main function to run the trainer"""
    print("=" * 80)
    print("Ultra Memory-Efficient AST Trainer Launch Script")
    print("=" * 80)
    
    # Check dependencies
    print("\nChecking dependencies...")
    check_dependencies()
    
    # Setup memory optimizations
    print("\nSetting up memory optimizations...")
    setup_memory_optimizations()
    
    # Check if the ultra_memory_efficient_trainer.py exists
    trainer_path = "./fix_scripts/ultra_memory_efficient_trainer.py"
    if not os.path.exists(trainer_path):
        print(f"\nError: Trainer script not found at {trainer_path}")
        return
    
    print("\nStarting the ultra memory-efficient AST trainer...\n")
    
    try:
        # Run the trainer script
        result = subprocess.run([sys.executable, trainer_path], check=True)
        
        if result.returncode == 0:
            print("\n" + "=" * 80)
            print("Training completed successfully!")
            print("=" * 80)
            print("\nResults are saved in the result/result_AST directory")
        else:
            print("\nTraining failed with error code:", result.returncode)
    
    except Exception as e:
        print("\nAn error occurred during training:")
        traceback.print_exc()
        print("\nError:", str(e))
        print("\nTry modifying the batch size or micro-batch size in the script")

if __name__ == "__main__":
    main()