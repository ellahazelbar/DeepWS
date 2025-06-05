import os
import shutil
import json
import cv2
import numpy as np
from tqdm import tqdm

def convert_windows_path_to_wsl(windows_path):
    """Convert Windows path to WSL path"""
    # Remove drive letter and convert backslashes to forward slashes
    path = windows_path.replace('\\', '/')
    if ':' in path:
        drive_letter = path[0].lower()
        path = f"/mnt/{drive_letter}{path[2:]}"
    return path

def process_wlasl_dataset(source_dir, target_dir):
    """
    Process the WLASL dataset and organize it for our model.
    
    Args:
        source_dir (str): Path to the downloaded WLASL dataset
        target_dir (str): Path to the target directory in our project
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Convert Windows path to WSL path
    source_dir_wsl = convert_windows_path_to_wsl(source_dir)
    
    # Load the WLASL metadata
    metadata_path = os.path.join(source_dir_wsl, 'WLASL_v0.3.json')
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Process each sign
    for sign_data in tqdm(metadata, desc="Processing signs"):
        gloss = sign_data['gloss']  # The English word for the sign
        
        # Create directory for this sign
        sign_dir = os.path.join(target_dir, gloss)
        os.makedirs(sign_dir, exist_ok=True)
        
        # Process each instance of the sign
        for instance in sign_data['instances']:
            video_id = instance['video_id']
            video_path = os.path.join(source_dir_wsl, 'videos', f'{video_id}.mp4')
            
            if not os.path.exists(video_path):
                print(f"Warning: Video {video_id} not found")
                continue
            
            # Copy the video to our organized structure
            target_path = os.path.join(sign_dir, f'{video_id}.mp4')
            shutil.copy2(video_path, target_path)

def main():
    # Source directory (where the dataset was downloaded)
    source_dir = r"C:\Users\naomi\.cache\kagglehub\datasets\risangbaskoro\wlasl-processed\versions\5"
    
    # Target directory in our project
    target_dir = os.path.join('data', 'raw')
    
    print(f"Processing WLASL dataset from {source_dir}")
    print(f"Organizing data into {target_dir}")
    
    process_wlasl_dataset(source_dir, target_dir)
    print("Dataset processing complete!")

if __name__ == '__main__':
    main() 