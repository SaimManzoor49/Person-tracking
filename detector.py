# detector.py
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
import numpy as np
from config import ReIDConfig

class PersonDetector:
    def __init__(self, config: ReIDConfig):
        self.config = config
        self.yolo = YOLO(config.yolo_model)
        self.extractor = FeatureExtractor(
            model_name=config.reid_model,
            device=config.device
        )

    def extract_features(self, frame, bbox):
        try:
            x1, y1, x2, y2 = map(int, bbox)
            height, width = y2 - y1, x2 - x1
            
            if height < self.config.min_bbox_size or width < self.config.min_bbox_size:
                return None
                
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
                
            features = self.extractor(crop)
            features_np = features.cpu().numpy().flatten()
            features_normalized = features_np / np.linalg.norm(features_np)
            return features_normalized
        except Exception as e:
            print(f"Feature extraction failed: {str(e)}")
            return None

    def detect(self, frame):
        detections = self.yolo(frame, classes=0)[0]
        
        valid_detections = []
        for det in detections.boxes.data:
            x1, y1, x2, y2, conf, _ = det
            if conf < self.config.detection_conf_threshold:
                continue
            
            bbox = [x1, y1, x2, y2]
            features = self.extract_features(frame, bbox)
            if features is not None:
                valid_detections.append((bbox, features))
                
        return valid_detections