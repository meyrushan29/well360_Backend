import os
import sys
import json
import time
import base64
import uuid
import traceback
import numpy as np
from io import BytesIO
import datetime as dt_module

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pydantic import BaseModel
from typing import Optional

from core.models import User, MentalHealthAnalysis
from core.deps import get_current_user
from core.database import get_db

router = APIRouter(
    prefix="/mental-health",
    tags=["Mental Health"]
)

# =====================================================
# PATH CONFIG
# =====================================================
MENTAL_H_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Mental-H", "Emotional")
AUDIO_MODEL_DIR = os.path.join(MENTAL_H_DIR, "AudioModel")
VIDEO_RECOMMENDATIONS_FILE = os.path.join(MENTAL_H_DIR, "recommendations.json")
AUDIO_RECOMMENDATIONS_FILE = os.path.join(AUDIO_MODEL_DIR, "recommendationA.json")

# Emotion classes for the CNN model
EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Stress-related emotions
STRESS_EMOTIONS = ["sad", "angry", "fear", "disgust"]

# =====================================================
# LAZY-LOADED MODELS (loaded on first use)
# =====================================================
_video_model = None
_audio_model = None


def _get_video_model():
    """Lazy-load the face emotion CNN model."""
    global _video_model
    if _video_model is not None:
        return _video_model

    try:
        import torch
        from torchvision import transforms

        # Add Mental-H/Emotional to sys.path to import the model
        if MENTAL_H_DIR not in sys.path:
            sys.path.insert(0, MENTAL_H_DIR)

        from other import EmotionCNN
        from config import DEVICE

        quantized_path = os.path.join(MENTAL_H_DIR, "emotion_model_quantized.pth")
        model_path = os.path.join(MENTAL_H_DIR, "emotion_model.pth")
        
        # Prefer quantized
        final_path = model_path
        use_quantized = False
        if os.path.exists(quantized_path):
             final_path = quantized_path
             use_quantized = True
             print(f"[Mental-H] Loading optimized model: {quantized_path}")
        elif not os.path.exists(model_path):
            raise FileNotFoundError(f"Video emotion model not found at {model_path}")

        model = EmotionCNN()
        
        if use_quantized:
             # Apply quantization structure before loading
             model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
             
        model.load_state_dict(torch.load(final_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        _video_model = {
            "model": model,
            "transform": transform,
            "device": DEVICE,
        }
        print("[Mental-H] Video emotion model loaded successfully")
        return _video_model

    except Exception as e:
        print(f"[Mental-H] Error loading video model: {e}")
        traceback.print_exc()
        return None


def _get_audio_model():
    """Lazy-load the audio emotion Random Forest model."""
    global _audio_model
    if _audio_model is not None:
        return _audio_model

    try:
        import joblib

        if AUDIO_MODEL_DIR not in sys.path:
            sys.path.insert(0, AUDIO_MODEL_DIR)

        model_path = os.path.join(AUDIO_MODEL_DIR, "Emotion_Model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Audio emotion model not found at {model_path}")

        model = joblib.load(model_path)

        _audio_model = {
            "model": model,
            "emotions": ['Fear', 'Angry', 'Disgust', 'Happy', 'Neutral', 'Sad', 'Surprise']
        }
        print("[Mental-H] Audio emotion model loaded successfully")
        return _audio_model

    except Exception as e:
        print(f"[Mental-H] Error loading audio model: {e}")
        traceback.print_exc()
        return None


def _load_recommendations(file_path):
    """Load recommendations JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("base_phrases", {})
    except Exception:
        return {}


def _get_recommendation_for_emotion(emotion: str, source: str = "video"):
    """Get recommendations for a given emotion."""
    rec_file = VIDEO_RECOMMENDATIONS_FILE if source == "video" else AUDIO_RECOMMENDATIONS_FILE
    recommendations = _load_recommendations(rec_file)

    # Map emotion labels to recommendation keys
    key_map = {
        "angry": "angry",
        "disgust": "disgusted",
        "fear": "fearful",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
        "surprise": "surprised",
    }

    key = key_map.get(emotion.lower(), emotion.lower())
    return recommendations.get(key, ["Stay calm and take care of yourself."])


def _save_video_emotion(db: Session, user_id: int, emotion: str, confidence: float = 0.0, faces_detected: int = 0):
    """Save emotion result to database."""
    entry = MentalHealthAnalysis(
        user_id=user_id,
        emotion=emotion,
        confidence=confidence,
        source="video",
        faces_detected=faces_detected
    )
    db.add(entry)
    db.commit()


def _save_audio_emotion(db: Session, user_id: int, emotion: str, confidence: float = 0.0, tone: str = None, energy: str = None):
    """Save emotion result to database."""
    entry = MentalHealthAnalysis(
        user_id=user_id,
        emotion=emotion,
        confidence=confidence,
        source="audio",
        tone=tone,
        energy=energy
    )
    db.add(entry)
    db.commit()


def _get_last_emotion(db: Session, user_id: int, source: str = "video"):
    """Get last detected emotion from DB."""
    entry = db.query(MentalHealthAnalysis).filter(
        MentalHealthAnalysis.user_id == user_id,
        MentalHealthAnalysis.source == source
    ).order_by(desc(MentalHealthAnalysis.timestamp)).first()
    
    if entry:
        return entry.emotion
    return None


def _compute_stress_from_history(db: Session, user_id: int, window: int = 20):
    """Compute stress level from recent emotion history using simple heuristic."""
    entries = db.query(MentalHealthAnalysis).filter(
        MentalHealthAnalysis.user_id == user_id
    ).order_by(desc(MentalHealthAnalysis.timestamp)).limit(window).all()

    if not entries:
        return {"stress_probability": 0.0, "stress_level": "Low", "emotions_analyzed": 0}

    # Reverse to chronological order for analysis if needed, but for count it doesn't matter
    emotions = [e.emotion.lower() for e in entries if e.emotion]
    
    if not emotions:
        return {"stress_probability": 0.0, "stress_level": "Low", "emotions_analyzed": 0}

    stress_count = sum(1 for e in emotions if e in STRESS_EMOTIONS)
    total = len(emotions)
    stress_prob = stress_count / total if total > 0 else 0.0

    if stress_prob >= 0.6:
        level = "High"
    elif stress_prob >= 0.3:
        level = "Moderate"
    else:
        level = "Low"

    return {
        "stress_probability": round(stress_prob, 2),
        "stress_level": level,
        "emotions_analyzed": total,
        "stress_emotions_count": stress_count,
    }


def _get_emotion_history(db: Session, user_id: int, source: str = "video", limit: int = 50):
    """Get recent emotion history with timestamps."""
    entries = db.query(MentalHealthAnalysis).filter(
        MentalHealthAnalysis.user_id == user_id,
        MentalHealthAnalysis.source == source
    ).order_by(desc(MentalHealthAnalysis.timestamp)).limit(limit).all()
    
    return [
        {
            "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "emotion": e.emotion,
            "confidence": e.confidence
        }
        for e in reversed(entries) # Return oldest to newest for charts
    ]


# =====================================================
# REQUEST MODELS
# =====================================================
class FrameAnalysisRequest(BaseModel):
    image_base64: str  # Base64 encoded face image


class AudioAnalysisRequest(BaseModel):
    audio_base64: str  # Base64 encoded audio file
    filename: Optional[str] = "upload.wav"


# =====================================================
# ENDPOINTS
# =====================================================

@router.get("/status")
def get_status():
    """Check module status and model availability."""
    video_available = os.path.exists(os.path.join(MENTAL_H_DIR, "emotion_model.pth"))
    audio_available = os.path.exists(os.path.join(AUDIO_MODEL_DIR, "Emotion_Model.pkl"))

    return {
        "status": "Module Active",
        "video_model_available": video_available,
        "audio_model_available": audio_available,
        "features": [
            "face_emotion_detection",
            "audio_emotion_detection",
            "stress_prediction",
            "recommendations"
        ]
    }


@router.post("/predict/face")
def predict_face_emotion(
    request: FrameAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Predict emotion from a face image (base64 encoded)."""
    try:
        import torch
        from PIL import Image
        import cv2

        model_data = _get_video_model()
        if model_data is None:
            raise HTTPException(status_code=503, detail="Video emotion model not available")

        model = model_data["model"]
        transform = model_data["transform"]
        device = model_data["device"]

        # Decode base64 image
        try:
            img_bytes = base64.b64decode(request.image_base64)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

        # Convert to numpy for face detection
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Detect faces
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            # Try to predict on the whole image as a face
            face_pil = img
            tensor = transform(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                _, pred = torch.max(output, 1)

            emotion = EMOTION_CLASSES[pred.item()]
            confidence = probabilities[pred.item()].item()

            _save_video_emotion(db, current_user.id, emotion, confidence, 0)

            return {
                "emotion": emotion,
                "confidence": round(confidence, 2),
                "faces_detected": 0,
                "note": "No face detected, analyzed full image",
                "recommendations": _get_recommendation_for_emotion(emotion, "video")
            }

        # Analyze the first/largest face
        results = []
        for (x, y, fw, fh) in faces:
            face = img_np[y:y + fh, x:x + fw]
            face_pil = Image.fromarray(face)
            tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                _, pred = torch.max(output, 1)

            emotion = EMOTION_CLASSES[pred.item()]
            confidence = probabilities[pred.item()].item()

            results.append({
                "emotion": emotion,
                "confidence": round(confidence, 2),
                "bbox": {"x": int(x), "y": int(y), "w": int(fw), "h": int(fh)}
            })

        # Use dominant face (first detected)
        primary = results[0]
        _save_video_emotion(db, current_user.id, primary["emotion"], primary["confidence"], len(faces))

        return {
            "emotion": primary["emotion"],
            "confidence": primary["confidence"],
            "faces_detected": len(faces),
            "all_faces": results,
            "recommendations": _get_recommendation_for_emotion(primary["emotion"], "video")
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Mental-H] Face prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/audio")
async def predict_audio_emotion(
    audio: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Predict emotion from an audio file upload."""
    try:
        import soundfile as sf

        model_data = _get_audio_model()
        if model_data is None:
            raise HTTPException(status_code=503, detail="Audio emotion model not available")

        model = model_data["model"]
        emotions_list = model_data["emotions"]

        # Ensure audio model dir is in path for imports
        if AUDIO_MODEL_DIR not in sys.path:
            sys.path.insert(0, AUDIO_MODEL_DIR)

        import Audio_feature_extraction as Afe

        # Save temp file
        os.makedirs("temp", exist_ok=True)
        temp_path = f"temp/{uuid.uuid4()}_{audio.filename}"

        try:
            contents = await audio.read()
            with open(temp_path, "wb") as f:
                f.write(contents)

            # Extract features
            with sf.SoundFile(temp_path) as audio_file:
                waveform = audio_file.read(dtype="float32")
                sample_rate = audio_file.samplerate

                chroma = Afe.feature_chromagram(waveform, sample_rate)
                mel = Afe.feature_melspectrogram(waveform, sample_rate)
                mfcc = Afe.feature_mfcc(waveform, sample_rate)
                features = np.hstack((chroma, mel, mfcc))

            # Predict
            prediction = model.predict([features])
            predicted_idx = int(prediction[0]) - 1
            if 0 <= predicted_idx < len(emotions_list):
                predicted_emotion = emotions_list[predicted_idx]
            else:
                predicted_emotion = "Neutral"

            # Get confidence (probability) if available
            confidence = 0.0
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba([features])
                    confidence = float(np.max(proba))
                except Exception:
                    confidence = 0.75  # Fallback

            # Determine tone and energy from audio features
            mfcc_mean = float(np.mean(mfcc))
            mel_mean = float(np.mean(mel))

            if mel_mean > 0.5:
                energy = "High"
            elif mel_mean > 0.2:
                energy = "Moderate"
            else:
                energy = "Low"

            if mfcc_mean > 0:
                tone = "Bright"
            elif mfcc_mean > -5:
                tone = "Steady"
            else:
                tone = "Deep"

            # Save to history
            _save_audio_emotion(db, current_user.id, predicted_emotion, confidence, tone, energy)

            return {
                "emotion": predicted_emotion,
                "confidence": round(confidence, 2),
                "tone": tone,
                "energy": energy,
                "recommendations": _get_recommendation_for_emotion(predicted_emotion, "audio")
            }

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Mental-H] Audio prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/last-emotion")
def get_last_emotion(
    source: str = "video",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the last detected emotion from history."""
    emotion = _get_last_emotion(db, current_user.id, source)

    if emotion is None:
        return {"has_previous": False, "emotion": None, "recommendations": []}

    return {
        "has_previous": True,
        "emotion": emotion,
        "recommendations": _get_recommendation_for_emotion(emotion, source)
    }


@router.get("/stress")
def get_stress_analysis(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get stress analysis from recent emotion history."""
    video_stress = _compute_stress_from_history(db, current_user.id)
    history = _get_emotion_history(db, current_user.id, limit=50)

    # Compute stress trend data (windowed stress probability)
    stress_trend = []
    window_size = 6
    if len(history) >= window_size:
        # history is returned in chronological order by _get_emotion_history now
        # so we can just iterate
        for i in range(window_size, len(history) + 1):
            window = history[i - window_size:i]
            emotions_in_window = [h["emotion"].lower() for h in window]
            stress_count = sum(1 for e in emotions_in_window if e in STRESS_EMOTIONS)
            stress_prob = stress_count / window_size
            stress_trend.append({
                "index": i - window_size,
                "stress_probability": round(stress_prob * 100, 1),  # As percentage
            })

    # Compute dominant emotion from recent history
    if history:
        from collections import Counter
        emotion_counts = Counter(h["emotion"] for h in history)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
    else:
        dominant_emotion = "Unknown"

    return {
        **video_stress,
        "dominant_emotion": dominant_emotion,
        "stress_trend": stress_trend,
        "history": history[-20:]  # Last 20 entries
    }


@router.get("/history")
def get_emotion_history_endpoint(
    source: str = "video",
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get emotion detection history."""
    history = _get_emotion_history(db, current_user.id, source, limit)
    return {"source": source, "count": len(history), "history": history}


@router.post("/predict/video")
async def predict_video_emotion(
    video: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Predict emotion from an uploaded video file. Processes frames and returns dominant emotion."""
    try:
        import torch
        import cv2
        from PIL import Image
        from collections import Counter

        model_data = _get_video_model()
        if model_data is None:
            raise HTTPException(status_code=503, detail="Video emotion model not available")

        model = model_data["model"]
        transform = model_data["transform"]
        device = model_data["device"]

        # Save temp file
        os.makedirs("temp", exist_ok=True)
        temp_path = f"temp/{uuid.uuid4()}_{video.filename}"

        try:
            contents = await video.read()
            with open(temp_path, "wb") as f:
                f.write(contents)

            # Open video with OpenCV
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open video file. Ensure it is a valid video.")

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            duration_sec = total_frames / fps if fps > 0 else 0

            # Sample frames (target ~60 frames max for better accuracy)
            sample_interval = max(1, total_frames // 60)
            
            # Weighted emotion scores
            emotion_scores = Counter()
            all_emotions = [] # Keep for breakdown count
            frame_results = []
            
            frame_idx = 0
            frames_processed = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                    if len(faces) > 0:
                        # Find largest face (most likely the user)
                        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                        x, y, fw, fh = largest_face
                        
                        face = frame[y:y + fh, x:x + fw]
                        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                        tensor = transform(face_pil).unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = model(tensor)
                            probabilities = torch.softmax(output, dim=1)[0]
                            prob, pred = torch.max(output, 1)

                        confidence = prob.item()
                        
                        # Only count confident predictions (> 40%)
                        if confidence > 0.4:
                            emotion = EMOTION_CLASSES[pred.item()]
                            
                            # Weight the vote by confidence
                            emotion_scores[emotion] += confidence
                            all_emotions.append(emotion)

                            frame_results.append({
                                "frame": frame_idx,
                                "time_sec": round(frame_idx / fps, 1) if fps > 0 else 0,
                                "emotion": emotion,
                                "confidence": round(confidence, 2),
                            })
                            
                            frames_processed += 1

                frame_idx += 1

            cap.release()

            if not all_emotions:
                return {
                    "emotion": "Unknown",
                    "confidence": 0.0,
                    "faces_detected": 0,
                    "total_frames": total_frames,
                    "frames_analyzed": frames_processed,
                    "duration_sec": round(duration_sec, 1),
                    "note": "No clear faces detected in the video",
                    "emotion_breakdown": {},
                    "frame_results": [],
                    "recommendations": ["Try uploading a clearer video with your face visible and good lighting."]
                }

            # Compute dominant emotion based on accumulated confidence scores
            dominant_emotion, total_score = emotion_scores.most_common(1)[0]
            
            # Calculate an aggregate confidence (average confidence for the dominant emotion)
            dominant_count = all_emotions.count(dominant_emotion)
            avg_confidence = total_score / dominant_count if dominant_count > 0 else 0.0
            
            # Boost confidence for display slightly if it's consistent
            display_confidence = min(0.99, avg_confidence * 1.1)

            # Confidence level label
            if display_confidence >= 0.75:
                confidence_label = "High"
            elif display_confidence >= 0.5:
                confidence_label = "Medium"
            else:
                confidence_label = "Low"

            # Emotion breakdown (percentages based on counts)
            total_detections = len(all_emotions)
            emotion_breakdown = {}
            raw_counts = Counter(all_emotions)
            
            for emo, count in raw_counts.most_common():
                emotion_breakdown[emo] = {
                    "count": count,
                    "percentage": round((count / total_detections) * 100, 1)
                }

            # Save to history
            _save_video_emotion(db, current_user.id, dominant_emotion, display_confidence, total_detections)

            return {
                "emotion": dominant_emotion,
                "confidence": round(display_confidence, 2),
                "confidence_label": confidence_label,
                "faces_detected": total_detections,
                "total_frames": total_frames,
                "frames_analyzed": frames_processed,
                "duration_sec": round(duration_sec, 1),
                "emotion_breakdown": emotion_breakdown,
                "frame_results": frame_results[-50:],  # Last 50 frame results
                "recommendations": _get_recommendation_for_emotion(dominant_emotion, "video")
            }

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Mental-H] Video prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{emotion}")
def get_recommendations(
    emotion: str,
    source: str = "video",
):
    """Get recommendations for a specific emotion (no auth required)."""
    recs = _get_recommendation_for_emotion(emotion, source)
    return {
        "emotion": emotion,
        "source": source,
        "recommendations": recs
    }
