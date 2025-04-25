import os
import shutil
from pathlib import Path

def reorganize_dataset(dataset_path, delete_original=True):
    # Create the target directories if they don't exist
    normal_dir = os.path.join(dataset_path, 'normal')
    abnormal_dir = os.path.join(dataset_path, 'abnormal')
    
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(abnormal_dir, exist_ok=True)
    
    # Keep track of directories to delete later
    dirs_to_delete = set()
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(dataset_path):
        # Skip the newly created normal and abnormal directories
        if root == dataset_path or root == normal_dir or root == abnormal_dir:
            continue
        
        path_parts = Path(root).parts
        dataset_parts = Path(dataset_path).parts
        
        # Get the relative path from the dataset directory
        rel_path_parts = path_parts[len(dataset_parts):]
        
        # Check if this is a normal or abnormal directory
        if 'normal' in rel_path_parts and 'abnormal' not in rel_path_parts:
            target_dir = normal_dir
            category = 'normal'
        elif 'abnormal' in rel_path_parts:
            target_dir = abnormal_dir
            category = 'abnormal'
        else:
            # Skip directories that are neither normal nor abnormal
            continue
        
        # Get parent directories for naming
        parent_dirs = [part for part in rel_path_parts if part != category]
        
        # Add directories to delete list
        if delete_original and len(rel_path_parts) > 0:
            top_level_dir = os.path.join(dataset_path, rel_path_parts[0])
            dirs_to_delete.add(top_level_dir)
        
        # Process each wav file
        for file in files:
            if file.endswith('.wav'):
                # Create the new filename
                new_name = f"{category}_{'_'.join(parent_dirs)}-{file}" if parent_dirs else f"{category}-{file}"
                
                # Source and destination paths
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_dir, new_name)
                
                # Copy the file
                print(f"Copying {src_file} to {dst_file}")
                shutil.copy2(src_file, dst_file)
    
    # Delete original directories if requested
    if delete_original:
        for dir_to_delete in sorted(dirs_to_delete, key=len, reverse=True):
            if os.path.exists(dir_to_delete) and dir_to_delete not in [normal_dir, abnormal_dir]:
                print(f"Removing directory: {dir_to_delete}")
                shutil.rmtree(dir_to_delete)

if __name__ == "__main__":
    # Set the path to your dataset
    dataset_path = "dataset"  # Change this to your dataset path if needed
    
    # Set to True if you want to delete the original folders after copying
    delete_original = True
    
    reorganize_dataset(dataset_path, delete_original)
    print("Dataset reorganization complete!")