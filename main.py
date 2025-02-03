# main.py
import cv2
from pathlib import Path
from config import ReIDConfig
from feature_extractor import FeatureManager
from detector import PersonDetector
from tracker import PersonTracker
from visualizer import Visualizer
def main():
    # Configuration
    config = ReIDConfig(
        yolo_model='./models/yolov8n.pt',
        reid_model='osnet_ain_x1_0',
        device='cpu',
        waiting_frames=2,
        max_features_per_id=10,
        min_bbox_size=100,
        iou_threshold=0.9,
        feature_match_threshold=0.8,
        max_frames_missing=40
    )
    
    # Initialize components
    feature_manager = FeatureManager(config)
    detector = PersonDetector(config)
    tracker = PersonTracker(config, feature_manager)
    
    # Video setup
    video_path = './videos/1.mp4'
    output_path = 'output.mp4'
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        valid_detections = detector.detect(frame)
        results = tracker.process_detections(valid_detections, frame_id)
        
        # Visualize results
        frame = Visualizer.draw_results(frame, results)
        
        # out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    feature_manager.save_database()

if __name__ == "__main__":
    main()