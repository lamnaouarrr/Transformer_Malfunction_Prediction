import os
import shutil
import sys
from pathlib import Path

def reorganize_audio_files(base_path):
    # Create base path object
    base_dir = Path(base_path)
    
    # Get the two main categories
    categories = ['normal', 'abnormal']
    
    for category in categories:
        category_path = base_dir / category
        
        # Check if the category directory exists
        if not category_path.exists():
            print(f"Directory {category_path} does not exist. Skipping.")
            continue
        
        # Create a temporary storage location to avoid conflicts during file movement
        temp_dir = base_dir / f"temp_{category}"
        temp_dir.mkdir(exist_ok=True)
        
        # First, move all files to temp directory
        moved_files = []
        for root, dirs, files in os.walk(category_path, topdown=False):
            # Skip the top-level category directory itself
            if Path(root) == category_path:
                continue
                
            for file in files:
                if file.endswith('.wav'):
                    # Get original file path
                    original_path = Path(root) / file
                    
                    # Calculate relative path from category directory
                    rel_path = Path(root).relative_to(category_path)
                    
                    # Create new filename with path information
                    path_parts = list(rel_path.parts)
                    new_filename = f"{category}_" + "_".join(path_parts) + "-" + file
                    
                    # Create destination path
                    temp_dest_path = temp_dir / new_filename
                    
                    try:
                        # Copy file to temp location
                        shutil.copy2(original_path, temp_dest_path)
                        moved_files.append((original_path, temp_dest_path))
                        print(f"Copied {original_path} to {temp_dest_path}")
                    except (PermissionError, OSError) as e:
                        print(f"Error copying {original_path}: {e}")
        
        # After copying all files, try deleting the originals
        for original_path, _ in moved_files:
            try:
                os.remove(original_path)
                print(f"Removed original file {original_path}")
            except (PermissionError, OSError) as e:
                print(f"Error removing original file {original_path}: {e}")
        
        # Now try to remove empty directories from bottom up
        for root, dirs, files in os.walk(category_path, topdown=False):
            # Skip the top-level category directory itself
            if Path(root) == category_path:
                continue
                
            if not os.listdir(root):  # Check if directory is empty
                try:
                    os.rmdir(root)
                    print(f"Removed empty directory {root}")
                except (PermissionError, OSError) as e:
                    print(f"Error removing directory {root}: {e}")
        
        # Move files from temp directory to category directory
        for file in temp_dir.iterdir():
            if file.is_file():
                try:
                    final_dest = category_path / file.name
                    shutil.move(file, final_dest)
                    print(f"Moved {file} to {final_dest}")
                except (PermissionError, OSError) as e:
                    print(f"Error moving {file} to final destination: {e}")
        
        # Remove temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory {temp_dir}")
        except (PermissionError, OSError) as e:
            print(f"Error removing temporary directory {temp_dir}: {e}")

if __name__ == "__main__":
    dataset_path = "dataset"
    
    # Check if command line argument is provided
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    print(f"Processing dataset at: {dataset_path}")
    reorganize_audio_files(dataset_path)
    print("Reorganization complete!")