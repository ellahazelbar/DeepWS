import torch
import torch.nn as nn
import torchvision.models as models

class ASLTranslator(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.5):
        super(ASLTranslator, self).__init__()
        
        # Load pretrained ResNet18 and remove the final fully connected layer
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=512,  # ResNet18's final feature size
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        
        # Reshape input for CNN
        c_in = x.view(batch_size * timesteps, C, H, W)
        
        # Extract features using CNN
        c_out = self.cnn(c_in)
        c_out = c_out.view(batch_size, timesteps, -1)
        
        # Process sequence with LSTM
        lstm_out, _ = self.lstm(c_out)
        
        # Use the last timestep's output for classification
        last_output = lstm_out[:, -1, :]
        
        # Final classification
        output = self.fc(last_output)
        
        return output

class ASLDataLoader:
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.transform = transform
        
    def load_video(self):
        """
        Load and preprocess video frames
        Returns:
            torch.Tensor: Processed video frames of shape (num_frames, channels, height, width)
        """
        import cv2
        import numpy as np
        
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                frame = self.transform(frame)
                
            frames.append(frame)
            
        cap.release()
        
        # Stack frames into a tensor
        frames = torch.stack(frames)
        return frames 