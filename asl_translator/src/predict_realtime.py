import os
import sys
import torch
import argparse
from torchvision import transforms
from models.cnn_lstm import ASLTranslator, ASLDataLoader
from models.keypoint_bilstm import KeypointBiLSTM
from data.piper import process_video
import numpy as np
import cv2
from matplotlib import pyplot as plt
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 

def predict_sign(frames, model_path, num_classes=3):
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
    model = KeypointBiLSTM(num_classes, 1662).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    frames = torch.FloatTensor([frames]).transpose(1,2).to(device)
    print(frames.shape)
    # Get prediction
    with torch.no_grad():
        outputs = model(frames)
        _, predicted = outputs.max(1)
    
    return ['change', 'cousin', 'trade'] [predicted.item()]

def mediapipe_detection(image, model): #mediapipe holistic model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #color conversion
    image.flags.writeable = False
    return model.process(image) #make prediction

def draw_landmarks(image, res):
    mp_drawing.draw_landmarks(image,res.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(0,110,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(0,256,121), thickness=1, circle_radius=1)) 
    mp_drawing.draw_landmarks(image,res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2)) 
    mp_drawing.draw_landmarks(image,res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,0,10), thickness=2, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(80,0,121), thickness=2, circle_radius=2)) 
    mp_drawing.draw_landmarks(image,res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,110,0), thickness=2, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(80,256,0), thickness=2, circle_radius=2)) 

def extract_keypoints(result):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]).flatten() if result.face_landmarks else np.zeros(1404)
    left_hand = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(63)
    right_hand = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(63)
    
    return np.concatenate([pose, right_hand, left_hand, face])

def main():
    frames = []
    cap = cv2.VideoCapture(0) #default bgr need rgb
    #set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: #detection is initial detection 
        while cap.isOpened():
            ret, frame = cap.read()
            
            frames.append(np.copy(frame))
            if 133 < len(frames):
                frames.pop(0)
            cv2.imshow('OpenCV Feed', frame)

            if cv2.waitKey(1000 // 30) != -1:
                break
        
        for i in range(len(frames)):
            cv2.imshow('OpenCV Feed', frames[i])
            res = mediapipe_detection(frames[i], holistic)
            frames[i] = extract_keypoints(res)
            cv2.waitKey(1000//30)
        while (len(frames) < 133):
            frames.append(np.zeros(1662))
            

    

    cap.release()
    cv2.destroyAllWindows()

    print(predict_sign(frames, 'asl_translator/src/models/best_model.pth'))

if __name__ == '__main__':
    main() 