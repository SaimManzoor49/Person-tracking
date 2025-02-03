from dataclasses import dataclass

@dataclass
class ReIDConfig:
    # Model paths and device
    yolo_model: str = 'yolov8n.pt'
    reid_model: str = 'osnet_ain_x1_0'
    device: str = 'cuda'
    
    # Detection parameters
    detection_conf_threshold: float = 0.3
    waiting_frames: int = 2
    max_features_per_id: int = 10
    min_bbox_size: int = 100
    iou_threshold: float = 0.9
    feature_match_threshold: float = 0.8
    max_frames_missing: int = 40
