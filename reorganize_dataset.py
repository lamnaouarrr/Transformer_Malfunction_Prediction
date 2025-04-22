import os
import shutil
from pathlib import Path

def reorganize_dataset(source_dir, target_dir):
    """
    Reorganizes a dataset by copying files from the original structure to a new structure
    with 'normal' and 'abnormal' as top-level categories while preserving subdirectories.
    
    Args:
        source_dir: Path to the source directory containing the original dataset
        target_dir: Path to the target directory where the reorganized dataset will be stored
    """
    # Create target directories if they don't exist
    normal_dir = os.path.join(target_dir, "normal")
    abnormal_dir = os.path.join(target_dir, "abnormal")
    
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(abnormal_dir, exist_ok=True)
    
    # Track stats
    processed = 0
    errors = 0
    
    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        # Skip if we're already in the target directory
        if target_dir in root:
            continue
        
        # Process only directories that contain 'normal' or 'abnormal'
        path_parts = Path(root).parts
        
        # Determine if this is a normal or abnormal directory
        if "normal" in path_parts and "abnormal" not in path_parts:
            is_abnormal = False
        elif "abnormal" in path_parts:
            is_abnormal = True
        else:
            # Skip directories that don't match our criteria
            continue
        
        # Determine the relative path to maintain the structure
        relative_path = os.path.relpath(root, source_dir)
        
        # Split the path to get the components we need
        path_components = Path(relative_path).parts
        
        # Find the common components to recreate the structure
        # We want to maintain everything except the 'normal'/'abnormal' distinction
        structure_path = []
        for component in path_components:
            if component != "normal" and component != "abnormal":
                structure_path.append(component)
        
        # Create the new path in the target directory
        if is_abnormal:
            target_subdir = os.path.join(abnormal_dir, *structure_path)
        else:
            target_subdir = os.path.join(normal_dir, *structure_path)
        
        # Create the directory structure
        os.makedirs(target_subdir, exist_ok=True)
        
        # Copy all audio files from this directory
        for file in files:
            if file.endswith('.wav'):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_subdir, file)
                
                try:
                    # Create parent directories if they don't exist
                    os.makedirs(os.path.dirname(target_file), exist_ok=True)
                    
                    # Copy the file
                    shutil.copy2(source_file, target_file)
                    print(f"Copied {source_file} to {target_file}")
                    processed += 1
                    
                except PermissionError:
                    print(f"Permission denied: Cannot copy {source_file} to {target_file}")
                    errors += 1
                except Exception as e:
                    print(f"Error copying {source_file}: {str(e)}")
                    errors += 1
    
    print(f"Dataset reorganization complete. Files processed: {processed}, Errors: {errors}")
    print(f"Check {target_dir} directory.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Reorganize dataset into normal and abnormal categories')
    parser.add_argument('--source', type=str, default='.', help='Source directory containing the original dataset')
    parser.add_argument('--target', type=str, default='./reorganized_dataset', help='Target directory for the reorganized dataset')
    
    args = parser.parse_args()
    
    # Make sure paths are absolute for clarity
    source_dir = os.path.abspath(args.source)
    target_dir = os.path.abspath(args.target)
    
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    
    reorganize_dataset(source_dir, target_dir)