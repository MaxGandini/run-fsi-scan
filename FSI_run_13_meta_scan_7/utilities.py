import re
import glob
import os
import cv2
from pathlib import Path
import numpy as np

def get_image_depth(image_name):
    match = re.search(r'-\s*([\d.]+)\s*m', image_name)
    if match:
        depth = float(match.group(1))
        return depth

def extract_num_from_name(filename, prefix: str) -> int:
    base = os.path.basename(str(filename))

    prefix_regex = re.escape(prefix)

    match = re.search(rf'{prefix_regex}[\s_]?(\d+)', base)

    return int(match.group(1)) if match else -1

def read_order_img(from_image, to_image, folder_path, prefix="Corte"):
    folder_path = str(folder_path)

    png_files = glob.glob(os.path.join(folder_path, f"{prefix}*.png"))
    jpg_files = glob.glob(os.path.join(folder_path, f"{prefix}*.jpg"))
    jpeg_files = glob.glob(os.path.join(folder_path, f"{prefix}*.jpeg"))

    image_files = png_files + jpg_files + jpeg_files

    image_files.sort(key=lambda f: extract_num_from_name(f, prefix))

    filtered_files = [
        file for file in image_files
        if from_image <= extract_num_from_name(file, prefix) <= to_image
    ]

    return filtered_files

def get_stack_around_from_list(image_list, index: int, n_image: int):

    start = max(index, 0)
    end = min(index + n_image, len(image_list))

    if start >= len(image_list):
        print("⚠️ Index out of range.")
        return None

    image_stack = image_list[start:end]

    gray_stack = [img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                  for img in image_stack]

    stack_array = np.stack(gray_stack, axis=0)

    return stack_array

def clean_images_from_folders(folder_paths, recursive=True, dry_run=False):
    image_extensions = {
        '.jpg', '.jpeg', '.png',
    }
    
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]
    
    results = {
        'deleted_files': [],
        'failed_deletions': [],
        'folders_processed': 0,
        'total_files_found': 0,
        'total_files_deleted': 0
    }
    
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' does not exist. Skipping...")
            continue
            
        if not os.path.isdir(folder_path):
            print(f"Warning: '{folder_path}' is not a directory. Skipping...")
            continue
        
        results['folders_processed'] += 1
        print(f"Processing folder: {folder_path}")
        
        # Use pathlib for easier file handling
        folder = Path(folder_path)
        
        # Find all files in the folder
        if recursive:
            files = folder.rglob('*')
        else:
            files = folder.glob('*')
        
        # Filter for image files
        image_files = [f for f in files if f.is_file() and f.suffix.lower() in image_extensions]
        results['total_files_found'] += len(image_files)
        
        print(f"Found {len(image_files)} image files")
        
        # Delete or show files
        for image_file in image_files:
            try:
                if dry_run:
                    print(f"Would delete: {image_file}")
                    results['deleted_files'].append(str(image_file))
                else:
                    image_file.unlink()  # Delete the file
                    print(f"Deleted: {image_file}")
                    results['deleted_files'].append(str(image_file))
                    results['total_files_deleted'] += 1
                    
            except Exception as e:
                error_msg = f"Failed to delete {image_file}: {str(e)}"
                print(f"Error: {error_msg}")
                results['failed_deletions'].append(error_msg)
    
    # Print summary
    print("\n" + "="*50)
    print("CLEANUP SUMMARY")
    print("="*50)
    print(f"Folders processed: {results['folders_processed']}")
    print(f"Image files found: {results['total_files_found']}")
    
    if dry_run:
        print(f"Files that would be deleted: {len(results['deleted_files'])}")
    else:
        print(f"Files successfully deleted: {results['total_files_deleted']}")
        print(f"Failed deletions: {len(results['failed_deletions'])}")
    
    return results

