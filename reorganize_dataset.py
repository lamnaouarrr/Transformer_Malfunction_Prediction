import os
import shutil
from pathlib import Path

def reorganize_dataset(source_dir, target_dir):
    """
    Reorganizes a dataset by moving files from a hierarchical structure to a flat structure
    with 'normal' and 'abnormal' categories.
    
    Args:
        source_dir: Path to the source directory containing the original dataset
        target_dir: Path to the target directory where the reorganized dataset will be stored
    """
    # Create target directories if they don't exist
    normal_dir = os.path.join(target_dir, "normal")
    abnormal_dir = os.path.join(target_dir, "abnormal")
    
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(abnormal_dir, exist_ok=True)
    
    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        # Check if this directory contains 'normal' or 'abnormal'
        path = Path(root)
        
        # Skip if we're already in the target directory
        if target_dir in root:
            continue
        
        # Determine if this is a normal or abnormal directory
        if "normal" in path.parts and "abnormal" not in path.parts[-1]:
            target_subdir = normal_dir
        elif "abnormal" in path.parts[-1]:
            target_subdir = abnormal_dir
        else:
            # Skip directories that don't match our criteria
            continue
            
        # Copy all audio files from this directory to the appropriate target directory
        for file in files:
            if file.endswith('.wav'):
                source_file = os.path.join(root, file)
                
                # Create a unique filename to avoid overwriting
                # Using the path components to create a unique name
                unique_parts = [p for p in path.parts if p not in ['normal', 'abnormal', '.']]
                unique_prefix = '_'.join(unique_parts)
                target_file = os.path.join(target_subdir, f"{unique_prefix}_{file}")
                
                shutil.copy2(source_file, target_file)
                print(f"Copied {source_file} to {target_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Reorganize dataset into normal and abnormal categories')
    parser.add_argument('--source', type=str, default='.', help='Source directory containing the original dataset')
    parser.add_argument('--target', type=str, default='./reorganized_dataset', help='Target directory for the reorganized dataset')
    
    args = parser.parse_args()
    
    reorganize_dataset(args.source, args.target)
    print(f"Dataset reorganization complete. Check {args.target} directory.")