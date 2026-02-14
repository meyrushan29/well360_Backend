import cv2
import torch
import time
import csv
import os
import json
from collections import Counter
from PIL import Image
from torchvision import transforms
import tkinter as tk
from tkinter import filedialog, messagebox

from other import EmotionCNN, ResNetEmotion
from collections import deque
from config import MODEL_PATH, DEVICE, EMOTION_CLASSES, IMG_SIZE
import numpy as np

# Override Model Path for Better Accuracy
MODEL_PATH = "best_resnet_model.pth"
IMG_SIZE = 224 # ResNet standard

# -------------------- PATH CONFIG --------------------
VIDEO_INPUT_DIR = r"C:\Users\Kavi\OneDrive\Desktop\ex"
VIDEO_OUTPUT_DIR = r"C:\Users\Kavi\OneDrive\Desktop\FaceEmoV2 pp1 -- ok (2)\FaceEmoV2 pp1 -- ok\savedEmoVideos"
CSV_FILE = "emotion_results.csv"
RECOMMENDATION_FILE = "recommendations.json"

os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# -------------------- FILE PICKER --------------------
def select_video_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )

# -------------------- LOAD RECOMMENDATIONS --------------------
def load_recommendations():
    with open(RECOMMENDATION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["base_phrases"]

recommendations = load_recommendations()

# -------------------- LOAD MODEL --------------------
# Try loading the better ResNet model first
try:
    print(f"Loading model from {MODEL_PATH}...")
    model = ResNetEmotion()
    # The saved model has keys like "model.conv1.weight", so we load it directly
    # or wrap it if needed. ResNetEmotion has self.model = resnet18.
    # The state_dict keys match the structure of ResNetEmotion (which has self.model).
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("Successfully loaded ResNetEmotion model.")
except Exception as e:
    print(f"Failed to load ResNet model: {e}")
    print("Falling back to simple EmotionCNN...")
    model = EmotionCNN()
    model.load_state_dict(torch.load("emotion_model.pth", map_location=DEVICE))

model.to(DEVICE)
model.eval()

# -------------------- TRANSFORM --------------------
# We need to set the transform dynamically based on which model loaded
if isinstance(model, ResNetEmotion):
    print("Using RGB transform for ResNet")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
else:
    print("Using Grayscale transform for EmotionCNN")
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)), # Old model size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

# -------------------- PREDICTION --------------------
# Smoothing buffer
prediction_buffer = deque(maxlen=10)

def predict_face(face_img):
    # Ensure image is RGB for ResNet, or convert to L for Grayscale if needed manually?
    # transforms.Grayscale() handles conversion if input is PIL Image.
    # However, if using ResNet, we need RGB.
    
    if isinstance(model, ResNetEmotion):
        if face_img.mode != 'RGB':
             face_img = face_img.convert('RGB')
    
    img = transform(face_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img)
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(output, dim=1)
        prob, pred = torch.max(probs, 1)
        
    # Threshold check (optional, but good for filtering weak predictions)
    # Only verify threshold for ResNet which is calibrated better
    if isinstance(model, ResNetEmotion) and prob.item() < 0.4:
        return "Neutral" # Fallback if unsure
        
    return EMOTION_CLASSES[pred.item()]

# -------------------- CONFIDENCE LEVEL --------------------
def get_confidence(count, total):
    ratio = count / total
    if ratio >= 0.7:
        return "High"
    elif ratio >= 0.4:
        return "Medium"
    else:
        return "Low"

# -------------------- SHOW RECOMMENDATIONS --------------------
recommendation_key_map = {
    "angry": "angry",
    "disgust": "disgusted",
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprise": "surprised"
}

def show_recommendations(emotion):
    print(f"\nDetected Emotion: {emotion}")
    print("Recommendations:")
    
    # Map emotion label (e.g., 'Fear') to JSON key (e.g., 'fearful')
    raw_key = emotion.lower()
    key = recommendation_key_map.get(raw_key, raw_key)
    
    if key in recommendations:
        for rec in recommendations[key][:5]:
            print(f"- {rec}")
    else:
        print("- No recommendation available for this emotion.")

# -------------------- MAIN PROCESS --------------------
def process_uploaded_video():
    video_path = select_video_file()
    if not video_path:
        print("No video selected")
        return

    video_name = os.path.basename(video_path)
    output_video = os.path.join(VIDEO_OUTPUT_DIR, f"processed_{video_name}")

    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    all_emotions = []

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "emotion", "video_name"])

    print("\nProcessing video...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, fw, fh) in faces:
            face = frame[y:y+fh, x:x+fw]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            # Predict
            raw_emotion = predict_face(face_pil)
            
            # Smooth prediction
            prediction_buffer.append(raw_emotion)
            # Get most common emotion in buffer
            emotion = Counter(prediction_buffer).most_common(1)[0][0]

            all_emotions.append(emotion)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, emotion, video_name])

            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 1)
            cv2.putText(frame, emotion, (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out.write(frame)

    cap.release()
    out.release()

    # -------------------- FINAL ANALYSIS --------------------
    if not all_emotions:
        print("No faces detected in the video")
        return

    emotion_counts = Counter(all_emotions)
    dominant_emotion, dominant_count = emotion_counts.most_common(1)[0]
    confidence = get_confidence(dominant_count, len(all_emotions))

    print("\n================ FINAL EMOTION ANALYSIS ================\n")
    print(f"Input Video      : {video_name}")
    print(f"Dominant Emotion : {dominant_emotion}")
    
    # Improved confidence calculation
    # If using smoothing, confidence is naturally higher.
    # We can also look at the ratio of the top 2 emotions.
    confidence = get_confidence(dominant_count, len(all_emotions))
    if confidence == "Low" and dominant_count / len(all_emotions) > 0.3:
         confidence = "Medium" # Boost slightly if significant enough
         
    print(f"Confidence       : {confidence}\n")

    show_recommendations(dominant_emotion)

    print("\n========================================================\n")
    print(f"Output video saved at:\n{output_video}")
    print(f"📄 CSV saved: {CSV_FILE}")

    # -------------------- PREDICT STRESS --------------------
    print("\nRun Stress Prediction...")
    import sys
    import subprocess
    subprocess.run([sys.executable, "Predict_Stress.py"])

# -------------------- RUN --------------------
# -------------------- PERSISTENCE LOGIC --------------------
def get_last_emotion():
    if not os.path.exists(CSV_FILE):
        return None
    try:
        with open(CSV_FILE, "r") as f:
            rows = list(csv.reader(f))
            if len(rows) > 1:
                return rows[-1][1] 
    except Exception as e:
        print(f"Error reading history: {e}")
    return None

def main():
    print("Mental Health Assessment - Video Analysis")
    
    last_emo = get_last_emotion()
    start_fresh = True
    
    if last_emo:
        root = tk.Tk()
        root.withdraw()
        ans = messagebox.askyesno("Follow-up", f"Last time you were feeling '{last_emo}'.\nAre you still feeling the same?")
        root.destroy()
        
        if ans:
             print(f"\nUser confirmed they are still feeling: {last_emo}")
             show_recommendations(last_emo)
             start_fresh = False
    
    if start_fresh:
        process_uploaded_video()

if __name__ == "__main__":
    main()
