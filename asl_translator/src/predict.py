import os
import sys
import torch
import argparse
from models.keypoint_bilstm import KeypointBiLSTM
from data.piper import process_video, KEYPOINTS_SIZE, FRAME_COUNT
import numpy as np

def predict_sign(video_path, model_path, num_classes=3):
    """
    Predict the ASL sign from a video file
    
    Args:
        video_path (str): Path to the input video file
        model_path (str): Path to the trained model weights
        num_classes (int): Number of classes in the model
    
    Returns:
        str: Predicted sign (letter)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = KeypointBiLSTM(num_classes, KEYPOINTS_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    frames = torch.FloatTensor([process_video(video_path, FRAME_COUNT)]).transpose(1,2).to(device)
    print(frames.shape)
    # Get prediction
    with torch.no_grad():
        outputs = model(frames)
        _, predicted = outputs.max(1)
    
    return ['change', 'cousin', 'trade'] [predicted.item()]

def main():
    parser = argparse.ArgumentParser(description='ASL Sign Prediction')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--model_path', type=str, default='asl_translator/src/models/best_model_gd.pth',
                      help='Path to the trained model weights')
    parser.add_argument('--num_classes', type=int, default=3,
                      help='Number of classes in the model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found")
        sys.exit(1)
    
    predicted_sign = predict_sign(args.video_path, args.model_path, args.num_classes)
    print(f"Predicted sign: {predicted_sign}")

if __name__ == '__main__':
    main() 