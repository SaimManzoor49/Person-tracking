# tracker.py
from collections import defaultdict
import numpy as np
from config import ReIDConfig
from feature_extractor import FeatureManager
from utils import calculate_iou
class PersonTracker:
    def __init__(self, config: ReIDConfig, feature_manager: FeatureManager):
        self.config = config
        self.feature_manager = feature_manager
        self.detection_history = {}
        self.waiting_detections = defaultdict(list)
        self.previous_detection_count = 0
        self.detection_change_frame = 0
        self.is_in_waiting_period = False

    def check_detection_count_change(self, current_count, frame_id):
        if current_count != self.previous_detection_count:
            self.is_in_waiting_period = True
            self.detection_change_frame = frame_id
            self.waiting_detections.clear()
            self.previous_detection_count = current_count
            return True
        return False

    def update_missing_tracks(self):
        for track_id in list(self.detection_history.keys()):
            self.detection_history[track_id]['frames_missing'] += 1
            if self.detection_history[track_id]['frames_missing'] > self.config.max_frames_missing:
                del self.detection_history[track_id]

    def process_detections(self, valid_detections, frame_id):
        results = []
        current_detection_count = len(valid_detections)
        detection_count_changed = self.check_detection_count_change(current_detection_count, frame_id)
        
        self.update_missing_tracks()

        if not self.feature_manager.feature_db and valid_detections:
            for bbox, features in valid_detections:
                new_id = self.feature_manager.get_next_id()
                self.feature_manager.update_feature_array(new_id, features)
                self.detection_history[new_id] = {
                    'bbox': bbox,
                    'features': features,
                    'frames_missing': 0
                }
                results.append((bbox, new_id))
            return results

        if self.is_in_waiting_period:
            results.extend(self._process_waiting_period(valid_detections, frame_id))
        else:
            results.extend(self._process_normal_period(valid_detections))

        return results

    def _process_waiting_period(self, valid_detections, frame_id):
        results = []
        for bbox, features in valid_detections:
            detection_key = tuple(map(int, bbox))
            self.waiting_detections[detection_key].append({
                'frame_id': frame_id,
                'features': features,
                'bbox': bbox
            })
            results.append((bbox, None))
            
        if frame_id - self.detection_change_frame >= self.config.waiting_frames:
            results.extend(self._process_waiting_detections())
            self.is_in_waiting_period = False
            
        return results

    def _process_waiting_detections(self):
        results = []
        for det_key, det_history in self.waiting_detections.items():
            if len(det_history) >= self.config.waiting_frames * 0.8:
                avg_features = np.mean([d['features'] for d in det_history], axis=0)
                avg_features = avg_features / np.linalg.norm(avg_features)
                
                matched_id, similarity = self.feature_manager.match_features(avg_features)
                if matched_id is None:
                    matched_id = self.feature_manager.get_next_id()
                
                self.feature_manager.update_feature_array(matched_id, avg_features)
                latest_detection = det_history[-1]
                self.detection_history[matched_id] = {
                    'bbox': latest_detection['bbox'],
                    'features': latest_detection['features'],
                    'frames_missing': 0
                }
                results.append((latest_detection['bbox'], matched_id))
        
        self.waiting_detections.clear()
        return results

    def _process_normal_period(self, valid_detections):
        results = []
        for bbox, features in valid_detections:
            best_iou = 0
            matched_id = None
            
            for track_id, track_info in self.detection_history.items():
                iou = calculate_iou(bbox, track_info['bbox'])
                if iou > self.config.iou_threshold and iou > best_iou:
                    best_iou = iou
                    matched_id = track_id
            
            if matched_id is not None:
                self._update_track(matched_id, bbox, features)
            else:
                matched_id = self._create_new_track(bbox, features)
                
            results.append((bbox, matched_id))
        return results

    def _update_track(self, track_id, bbox, features):
        self.detection_history[track_id] = {
            'bbox': bbox,
            'features': features,
            'frames_missing': 0
        }
        self.feature_manager.update_feature_array(track_id, features)

    def _create_new_track(self, bbox, features):
        matched_id, similarity = self.feature_manager.match_features(features)
        if matched_id is None:
            matched_id = self.feature_manager.get_next_id()
            
        self._update_track(matched_id, bbox, features)
        return matched_id