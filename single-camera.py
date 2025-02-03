import json
import logging
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from collections import defaultdict, deque
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PersonReID:
    def __init__(self, 
                 yolo_model='./models/yolov8n.pt', 
                 reid_model='osnet_ibn_x1_0', 
                 device='cuda',
                 waiting_frames=3,
                 max_features_per_id=90,
                 min_bbox_size=90,
                 iou_threshold=0.98,
                 feature_match_threshold=0.76,
                 max_frames_missing=40,
                 detection_confidence=0.4):
        
        self.device = device
        self.yolo = YOLO(yolo_model)
        self.extractor = FeatureExtractor(model_name=reid_model, device=device)
        
        # Tracking components
        self.next_id = 1
        self.detection_history = {}
        self.waiting_detections = defaultdict(list)
        self.feature_db = {}  # Stores clusters: {id: [{'centroid': ..., 'count': int}]}
        self.last_active = {}  # Last active frame for each ID
        self.db_path = Path('reid_database.json')
        
        # Configuration parameters
        self.WAITING_FRAMES = waiting_frames
        self.MAX_FEATURES_PER_ID = max_features_per_id
        self.MIN_BBOX_SIZE = min_bbox_size
        self.IOU_THRESHOLD = iou_threshold
        self.FEATURE_MATCH_THRESHOLD = feature_match_threshold
        self.MAX_FRAMES_MISSING = max_frames_missing
        self.DETECTION_CONFIDENCE = detection_confidence
        self.CLUSTER_THRESHOLD = 0.85  # Similarity threshold for clustering
        self.MAX_CLUSTERS_PER_ID = 5    # Max clusters per ID
        self.PRUNE_INTERVAL = 1000      # Prune database every N frames
        self.MAX_INACTIVE_FRAMES = 5000 # Remove IDs inactive for this many frames

        # State management
        self.previous_detection_count = 0
        self.detection_change_frame = 0
        self.is_in_waiting_period = False
        
        self.load_database()

    def load_database(self):
        """Robust database loading with validation"""
        try:
            if self.db_path.exists():
                with open(self.db_path, 'r') as f:
                    serialized_db = json.load(f)
                
                self.feature_db = {}
                for obj_id_str, clusters in serialized_db.items():
                    try:
                        obj_id = int(obj_id_str)
                        self.feature_db[obj_id] = [{
                            'centroid': np.array(c['centroid']),
                            'count': c['count']
                        } for c in clusters]
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Skipping invalid entry {obj_id_str}: {str(e)}")
                
                if self.feature_db:
                    self.next_id = max(self.feature_db.keys()) + 1
                else:
                    self.next_id = 1
                
                logging.info(f"Loaded database with {len(self.feature_db)} identities")
            else:
                self.feature_db = {}
                self.next_id = 1

        except Exception as e:
            logging.error(f"Error loading database: {str(e)}")
            self.feature_db = {}
            self.next_id = 1

        # Initialize last_active
        self.last_active = {k: 0 for k in self.feature_db.keys()}

    def save_database(self):
        """Atomic database saving with error handling"""
        try:
            temp_path = self.db_path.with_suffix('.tmp')
            serialized_db = {
                str(k): [{'centroid': c['centroid'].tolist(), 'count': c['count']} 
                        for c in clusters]
                for k, clusters in self.feature_db.items()
            }
            
            with open(temp_path, 'w') as f:
                json.dump(serialized_db, f, indent=2)
            
            temp_path.replace(self.db_path)
            logging.info(f"Database saved with {len(self.feature_db)} identities")
        
        except Exception as e:
            logging.error(f"Database save failed: {str(e)}")

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union with stability checks"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = box1_area + box2_area - intersection
        return intersection / (union + 1e-9)

    def check_bbox_size(self, bbox):
        """Check if bbox meets minimum size requirements"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        return h >= self.MIN_BBOX_SIZE and w >= self.MIN_BBOX_SIZE

    def batch_extract_features(self, frame, bboxes):
        """Batch feature extraction with augmentations"""
        valid_features = []
        crops = []
        augmented_indices = []
        valid_bboxes = []

        for idx, bbox in enumerate(bboxes):
            if not self.check_bbox_size(bbox):
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or min(crop.shape[:2]) < 10:
                continue
            
            # Generate augmented versions
            crops.append(crop)
            augmented_indices.append(idx)
            flipped = cv2.flip(crop, 1)
            crops.append(flipped)
            augmented_indices.append(idx)
            
            valid_bboxes.append(bbox)

        if crops:
            try:
                features = self.extractor(crops)
                features_np = features.cpu().numpy()
                
                # Average features from augmented versions
                features_dict = defaultdict(list)
                for i, idx in enumerate(augmented_indices):
                    feat = features_np[i] / np.linalg.norm(features_np[i])
                    features_dict[idx].append(feat)
                
                for idx in features_dict:
                    avg_feature = np.mean(features_dict[idx], axis=0)
                    avg_feature /= np.linalg.norm(avg_feature)
                    valid_features.append((valid_bboxes[idx], avg_feature))
            except Exception as e:
                logging.error(f"Feature extraction failed: {str(e)}")

        return valid_features

    def match_features(self, query_features, threshold=None):
        """Feature matching with dynamic threshold"""
        threshold = threshold or self.FEATURE_MATCH_THRESHOLD
        if not self.feature_db:
            return None, 0.0

        query_norm = query_features / np.linalg.norm(query_features)
        best_match = (None, threshold)
        
        for obj_id, clusters in self.feature_db.items():
            max_sim = max(
                np.dot(query_norm, cluster['centroid']) 
                for cluster in clusters
            )
            if max_sim > best_match[1]:
                best_match = (obj_id, max_sim)

        return best_match if best_match[0] else (None, best_match[1])

    def update_feature_store(self, obj_id, new_features, frame_id):
        """Update features with clustering"""
        new_norm = new_features / np.linalg.norm(new_features)
        
        if obj_id not in self.feature_db:
            self.feature_db[obj_id] = []
            self.last_active[obj_id] = frame_id
        
        clusters = self.feature_db[obj_id]
        best_sim, best_idx = -1, -1
        
        for i, cluster in enumerate(clusters):
            sim = np.dot(new_norm, cluster['centroid'])
            if sim > best_sim:
                best_sim, best_idx = sim, i

        if best_sim >= self.CLUSTER_THRESHOLD and best_idx != -1:
            # Merge with existing cluster
            cluster = clusters[best_idx]
            new_count = cluster['count'] + 1
            cluster['centroid'] = (cluster['centroid']*cluster['count'] + new_norm) / new_count
            cluster['count'] = new_count
        else:
            # Add new cluster
            clusters.append({'centroid': new_norm, 'count': 1})
            # Remove oldest if over limit
            if len(clusters) > self.MAX_CLUSTERS_PER_ID:
                del clusters[0]
        
        self.last_active[obj_id] = frame_id

    def prune_database(self, current_frame):
        """Remove inactive IDs"""
        threshold = current_frame - self.MAX_INACTIVE_FRAMES
        to_remove = [k for k, v in self.last_active.items() if v < threshold]
        
        for k in to_remove:
            del self.feature_db[k]
            del self.last_active[k]
        
        logging.info(f"Pruned {len(to_remove)} inactive IDs")

    def manage_tracking_states(self, current_count, frame_id):
        """Handle detection count changes and state transitions"""
        count_changed = current_count != self.previous_detection_count
        
        if count_changed:
            self.is_in_waiting_period = True
            self.detection_change_frame = frame_id
            self.waiting_detections.clear()
            self.previous_detection_count = current_count
            logging.debug(f"Detection count changed to {current_count}")
        
        return count_changed

    def process_frame(self, frame, frame_id):
        """Main processing pipeline with pruning"""
        if frame_id % self.PRUNE_INTERVAL == 0:
            self.prune_database(frame_id)
            
        results = []
        yolo_results = self.yolo(frame, classes=0)[0]
        
        # Process YOLO detections
        valid_detections = []
        bboxes = []
        
        for det in yolo_results.boxes.data:
            x1, y1, x2, y2, conf, _ = map(float, det)
            if conf < self.DETECTION_CONFIDENCE:
                continue
            
            bbox = [x1, y1, x2, y2]
            bboxes.append(bbox)

        # Batch feature extraction
        valid_detections = self.batch_extract_features(frame, bboxes)
        current_count = len(valid_detections)
        self.manage_tracking_states(current_count, frame_id)

        # Update missing frames count for all tracks
        for track_id in self.detection_history:
            self.detection_history[track_id]['frames_missing'] += 1

        # Cleanup stale tracks
        self.cleanup_stale_tracks()

        if not self.feature_db and valid_detections:
            return self.handle_initial_detections(valid_detections, frame_id)

        # Process based on current state
        if self.is_in_waiting_period:
            return self.process_waiting_period(valid_detections, frame_id)
        else:
            return self.process_normal_detections(valid_detections, frame_id)

    def handle_initial_detections(self, valid_detections, frame_id):  # Added frame_id parameter
        """Handle initial population of empty database"""
        results = []
        for bbox, features in valid_detections:
            if not self.check_bbox_size(bbox):
                continue
            new_id = self.next_id
            self.next_id += 1
            self.update_feature_store(new_id, features, frame_id)  # Added frame_id
            self.detection_history[new_id] = {
                'bbox': bbox,
                'features': features,
                'velocity': (0, 0),
                'frames_missing': 0
            }
            results.append((bbox, new_id))
        return results

    def process_waiting_period(self, valid_detections, frame_id):
        """Wait period processing with confidence threshold"""
        # Use higher threshold during stabilization
        temp_threshold = self.FEATURE_MATCH_THRESHOLD * 1.1
        temp_results = []
        
        for bbox, features in valid_detections:
            if not self.check_bbox_size(bbox):
                continue
            detection_key = tuple(map(int, bbox))
            self.waiting_detections[detection_key].append({
                'frame_id': frame_id,
                'features': features,
                'bbox': bbox
            })
            temp_results.append((bbox, None))

        if frame_id - self.detection_change_frame >= self.WAITING_FRAMES:
            self.is_in_waiting_period = False
            confirmed = []
            
            for det_key, history in self.waiting_detections.items():
                if len(history) >= max(2, int(self.WAITING_FRAMES * 0.5)):
                    try:
                        latest = history[-1]
                        if not self.check_bbox_size(latest['bbox']):
                            continue
                            
                        avg_features = np.mean([d['features'] for d in history], axis=0)
                        avg_features /= np.linalg.norm(avg_features)
                        
                        match_id, similarity = self.match_features(avg_features, threshold=temp_threshold)

                        if match_id is None:
                            match_id = self.next_id
                            self.next_id += 1
                            logging.info(f"New ID {match_id} created (similarity: {similarity:.2f})")
                        
                        self.update_feature_store(match_id, avg_features, frame_id)
                        self.detection_history[match_id] = {
                            'bbox': latest['bbox'],
                            'features': latest['features'],
                            'velocity': (0, 0),
                            'frames_missing': 0
                        }
                        confirmed.append((latest['bbox'], match_id))
                    except Exception as e:
                        logging.error(f"Error processing waiting detection: {str(e)}")
            
            self.waiting_detections.clear()
            return confirmed
        
        return temp_results

    def process_normal_detections(self, valid_detections,frame_id):
        """Process detections in normal tracking mode"""
        results = []
        
        for bbox, features in valid_detections:
            if not self.check_bbox_size(bbox):
                continue
                
            # IOU matching with motion prediction
            best_match = self.match_iou_with_prediction(bbox)
            
            # Feature matching fallback
            if best_match is None:
                best_match, similarity = self.match_features(features)
                if best_match is not None:
                    logging.debug(f"Feature match {best_match} ({similarity:.2f})")
            
            # New identity creation
            if best_match is None:
                best_match = self.next_id
                self.next_id += 1
                logging.info(f"New ID {best_match} created")
            
            # Update tracking information
            self.update_track(best_match, bbox, features, frame_id)  
            results.append((bbox, best_match))
        
        return results

    def match_iou_with_prediction(self, bbox):
        """IOU matching with velocity prediction"""
        if not self.check_bbox_size(bbox):
            return None
            
        best_iou = 0.0
        best_match = None
        
        for track_id, track in self.detection_history.items():
            predicted_bbox = self.predict_bbox(track['bbox'], track.get('velocity', (0, 0)))
            iou = self.calculate_iou(bbox, predicted_bbox)
            
            if iou > best_iou and iou > self.IOU_THRESHOLD:
                best_iou = iou
                best_match = track_id
        
        return best_match

    def predict_bbox(self, bbox, velocity):
        """Simple linear motion prediction"""
        dx, dy = velocity
        return [
            bbox[0] + dx,
            bbox[1] + dy,
            bbox[2] + dx,
            bbox[3] + dy
        ]

    def update_track(self, track_id, bbox, features,frame_id):
        """Update track with velocity calculation"""
        if not self.check_bbox_size(bbox):
            return
            
        prev_entry = self.detection_history.get(track_id, {})
        prev_bbox = prev_entry.get('bbox', bbox)
        
        # Calculate velocity (pixels/frame)
        velocity = (
            (bbox[0] + bbox[2] - prev_bbox[0] - prev_bbox[2]) / 2,
            (bbox[1] + bbox[3] - prev_bbox[1] - prev_bbox[3]) / 2)
        
        # Update track entry
        self.detection_history[track_id] = {
            'bbox': bbox,
            'features': features,
            'velocity': velocity,
            'frames_missing': 0
        }
        
        # Update feature storage
        self.update_feature_store(track_id, features, frame_id)

    def cleanup_stale_tracks(self):
        """Remove tracks that haven't been updated recently"""
        stale_ids = []
        for track_id, track in self.detection_history.items():
            if track['frames_missing'] > self.MAX_FRAMES_MISSING:
                stale_ids.append(track_id)
            elif not self.check_bbox_size(track['bbox']):
                track['frames_missing'] = self.MAX_FRAMES_MISSING + 1
                stale_ids.append(track_id)
        
        for track_id in stale_ids:
            del self.detection_history[track_id]
            logging.info(f"Removed stale track {track_id}")

def main():
    reid_system = PersonReID(
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    cap = cv2.VideoCapture(1)  # Use webcam
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = reid_system.process_frame(frame, frame_id)
        frame_id += 1

        # Visualization
        for bbox, obj_id in results:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0) if obj_id is not None else (0, 255, 255)
            label = f"ID: {obj_id}" if obj_id else "Detecting..."
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Person Re-ID', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    reid_system.save_database()

if __name__ == "__main__":
    main()