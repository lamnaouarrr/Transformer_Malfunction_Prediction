import os
import shutil
import sys
from pathlib import Path

def reorganize_audio_files(base_path, delete_originals=False):
    """
    Copy audio files from nested directories to the top-level normal and abnormal directories
    with appropriate naming, optionally attempting to delete originals if specified.
    
    Args:
        base_path: Path to the dataset directory
        delete_originals: Whether to attempt to delete original files (default: False)
    """
    # Create base path object
    base_dir = Path(base_path)
    
    # Get the two main categories
    categories = ['normal', 'abnormal']
    
    # Track statistics
    stats = {
        'copied': 0,
        'errors': 0
    }
    
    for category in categories:
        category_path = base_dir / category
        
        # Check if the category directory exists
        if not category_path.exists():
            print(f"Directory {category_path} does not exist. Skipping.")
            continue
        
        print(f"\nProcessing {category} files...")
        
        # Walk through all directories and files in this category
        for root, _, files in os.walk(category_path):
            # Skip processing the top-level category directory itself
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
                    
                    # Create destination path directly in category folder
                    dest_path = category_path / new_filename
                    
                    try:
                        # Copy file to destination
                        shutil.copy2(original_path, dest_path)
                        print(f"Copied: {new_filename}")
                        stats['copied'] += 1
                        
                        # Optionally attempt to delete original if requested
                        if delete_originals:
                            try:
                                os.remove(original_path)
                            except (PermissionError, OSError) as e:
                                # Just log deletion errors but continue
                                print(f"Note: Couldn't delete original {original_path}: {e}")
                    except (PermissionError, OSError) as e:
                        print(f"Error copying {original_path}: {e}")
                        stats['errors'] += 1
    
    print(f"\nSummary:")
    print(f"Files successfully copied: {stats['copied']}")
    print(f"Errors encountered: {stats['errors']}")
    print("\nNote: Original directory structure remains unchanged.")
    if not delete_originals:
        print("Original files were not deleted due to permission issues.")
        print("You can manually delete the nested directories after verifying the copied files.")

if __name__ == "__main__":
    # Default dataset path
    dataset_path = "dataset"
    
    # Check if command line argument is provided
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    print(f"Processing dataset at: {dataset_path}")
    
    # Run without attempting to delete originals
    reorganize_audio_files(dataset_path, delete_originals=False)
    
    print("\nReorganization complete!")