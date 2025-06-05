import os
import sys
import torch
import argparse
from torchvision import transforms
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_lstm import ASLTranslator, ASLDataLoader

def predict_sign(video_path, model_path, num_classes=26):
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
    model = ASLTranslator(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Prepare video
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    loader = ASLDataLoader(video_path, transform)
    frames = loader.load_video()
    frames = frames.unsqueeze(0).to(device)  # Add batch dimension
    
    # Get prediction
    with torch.no_grad():
        outputs = model(frames)
        _, predicted = outputs.max(1)
    
    # Convert prediction to letter
    predicted_letter = chr(predicted.item() + ord('A'))
    return predicted_letter

def main():
    parser = argparse.ArgumentParser(description='ASL Sign Prediction')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--model_path', type=str, default='../../models/best_model.pth',
                      help='Path to the trained model weights')
    parser.add_argument('--num_classes', type=int, default=26,
                      help='Number of classes in the model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found")
        sys.exit(1)
    
    try:
        predicted_sign = predict_sign(args.video_path, args.model_path, args.num_classes)
        print(f"Predicted sign: {predicted_sign}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 