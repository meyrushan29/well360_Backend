import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

class LipExtractor:
    def __init__(self, model_path=None):
        """
        Initialize LipExtractor with MediaPipe FaceLandmarker (new API).
        
        Args:
            model_path: Path to face_landmarker.task model file. 
                       If None, will attempt to download it.
        """
        # Detailed Lip Landmark Indices (Outer Boundary)
        self.LIP_OUTER_INDICES = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
            409, 270, 269, 267, 0, 37, 39, 40, 185
        ]
        # Inner boundary (optional, but good for precision)
        self.LIP_INNER_INDICES = [
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
            415, 310, 311, 312, 13, 82, 81, 80, 191
        ]
        
        self.ALL_LIP_INDICES = list(set(self.LIP_OUTER_INDICES + self.LIP_INNER_INDICES))
        
        # Download model if not provided
        if model_path is None:
            model_path = self._download_model()
        
        # Create FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
    
    def _download_model(self):
        """Download the face_landmarker.task model if not present."""
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "face_landmarker.task")
        
        if not os.path.exists(model_path):
            print("Downloading face_landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print(f"Model downloaded to {model_path}")
        
        return model_path

    def extract_lips(self, image: np.ndarray) -> Tuple[Optional[List[Tuple[int, int]]], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extracts lips from an image using MediaPipe Face Landmarker.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            - lip_landmarks: List of (x, y) coordinates
            - lip_mask: Binary mask of the lip region
            - lip_roi: Cropped lip image
        """
        h, w, _ = image.shape
        
        # Convert to RGB and create MediaPipe Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect face landmarks
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None, None, None

        # Get landmarks for the first face detected
        face_landmarks = detection_result.face_landmarks[0]
        
        # 1. Lip Landmark Coordinates
        lip_coords = []
        for idx in self.LIP_OUTER_INDICES:
            landmark = face_landmarks[idx]
            lip_coords.append((int(landmark.x * w), int(landmark.y * h)))
        
        # 2. Lip Segmentation Mask
        mask = np.zeros((h, w), dtype=np.uint8)
        points = np.array(lip_coords, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        
        # 3. Cropped Lip ROI
        # Find bounding box
        x, y, bw, bh = cv2.boundingRect(points)
        
        # Use mask to black out everything except lips
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Crop to bounding box
        lip_roi = masked_image[y:y+bh, x:x+bw]
        
        return lip_coords, mask, lip_roi

if __name__ == "__main__":
    # Test on a dummy image or real image if available
    extractor = LipExtractor()
    # test_img = cv2.imread("some_path.jpg")
    # coords, mask, roi = extractor.extract_lips(test_img)
    # if roi is not None:
    #     cv2.imshow("Lip ROI", roi)
    #     cv2.waitKey(0)

