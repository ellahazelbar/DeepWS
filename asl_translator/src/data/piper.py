import mediapipe as mp
import os
import cv2
from torchvision import transforms
import numpy as np 
import csv


transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
])

data_dir = 'asl_translator/src/data/processed'
output_dir = 'asl_translator/src/data/piped'
mp_holistic = mp.solutions.holistic #holistic model

max_frames = -1

def preprocess_video(input_path):
    global max_frames
    cap = cv2.VideoCapture(input_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if fps not in [24, 25, 30]:
        raise("Unsupported framerate")
    frames_multiplier = 1
    if fps == 24:
        frames_multiplier = 1.25
    elif fps == 25:
        frames_multiplier = 1.1666666666666
    framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    eventual_framecount = int(framecount * frames_multiplier)
    if max_frames < eventual_framecount:
        max_frames = eventual_framecount

def process_video(input_path, output_path, mp_model):
    cap = cv2.VideoCapture(input_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if fps not in [24, 25, 30]:
        raise("Unsupported framerate")
    
    repeat_freq = 100000000
    if fps == 24:
        repeat_freq = 4
    elif fps == 25:
        repeat_freq = 5
    repeat_counter = 0

    output = open(output_path, 'wb')
    frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = extract_keypoints(mp_model.process(frame))
        np.asarray(keypoints, dtype=np.float32).tofile(output)
        frames += 1
        repeat_counter += 1
        if (repeat_freq == repeat_counter):
            repeat_counter = 0
            np.asarray(keypoints, dtype=np.float32).tofile(output)
            frames += 1
    cap.release()
    pad = np.asarray(np.zeros(1662), dtype=np.float32)
    while (frames < max_frames):
        pad.tofile(output)
        frames += 1


def extract_keypoints(result):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]).flatten() if result.face_landmarks else np.zeros(1404)
    left_hand = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(63)
    right_hand = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(63)
    
    return np.concatenate([pose, right_hand, left_hand, face])

for class_name in sorted(os.listdir(data_dir)):            
    class_dir = os.path.join(data_dir, class_name)
    for video_file in os.listdir(class_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            input_path = os.path.join(class_dir, video_file)
            preprocess_video(input_path)
x = 3
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:#detection is initial detection 
    for class_name in sorted(os.listdir(data_dir)):            
        class_dir = os.path.join(data_dir, class_name)
        for video_file in os.listdir(class_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                noext, _ = os.path.splitext(video_file)
                input_path = os.path.join(class_dir, video_file)
                output_path = os.path.join(output_dir, class_name, noext + ".piped")
                process_video(input_path, output_path, holistic)
                    




