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
                 yolo_model='yolov8n-pose.pt',
                 reid_model='osnet_ain_x1_0',
                 device='cuda',
                 waiting_frames=9,
                 # MAX_FEATURES_PER_ID removed as it's not used in the current clustering logic
                 min_bbox_size=90,
                 track_iou_threshold=0.99,  # Renamed and default changed from 0.98
                 feature_match_threshold=0.74,
                 max_frames_missing=120,
                 detection_confidence=0.4,
                 part_confidence=0.4,
                 nms_pre_reid_iou_threshold=0.95): # New: NMS for initial detections

        self.device = device
        self.yolo = YOLO(yolo_model)
        self.extractor = FeatureExtractor(model_name=reid_model, device=device,model_path='./models/osnet_ain_x1_0_market1501_256x128_amsgrad_ep100_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth')

        self.next_id = 1
        self.detection_history = {}  # Stores current state of tracked IDs
        self.waiting_detections = defaultdict(list) # Stores detections during waiting period
        self.feature_db = {}  # Stores feature clusters for each ID
        self.last_active = {} # Stores last frame_id an ID was active
        self.db_path = Path('reid_database.json')

        self.WAITING_FRAMES = waiting_frames
        self.MIN_BBOX_SIZE = min_bbox_size
        self.TRACK_IOU_THRESHOLD = track_iou_threshold # Used for matching detections to tracks
        self.FEATURE_MATCH_THRESHOLD = feature_match_threshold
        self.MAX_FRAMES_MISSING = max_frames_missing
        self.DETECTION_CONFIDENCE = detection_confidence
        self.NMS_PRE_REID_IOU_THRESHOLD = nms_pre_reid_iou_threshold # NMS for YOLO outputs

        self.CLUSTER_THRESHOLD = feature_match_threshold+0.2 # For updating feature clusters
        self.MAX_CLUSTERS_PER_ID = 10
        self.PRUNE_INTERVAL = 10000 # Frames after which to prune DB
        self.MAX_INACTIVE_FRAMES = 5000 # Max frames an ID can be inactive before pruning
        self.PART_CONFIDENCE = part_confidence

        self.previous_detection_count = 0
        self.detection_change_frame = 0
        self.is_in_waiting_period = False

        self.load_database()

    def load_database(self):
        try:
            if self.db_path.exists():
                with open(self.db_path, 'r') as f:
                    serialized_db = json.load(f)

                self.feature_db = {}
                valid_ids_loaded = []
                for obj_id_str, clusters_data in serialized_db.items():
                    try:
                        obj_id = int(obj_id_str)
                        clusters = []
                        for c_data in clusters_data:
                            if 'centroid' in c_data and 'count' in c_data:
                                centroid = np.array(c_data['centroid'])
                                # Ensure centroid is 1D array (flatten if necessary)
                                if centroid.ndim > 1:
                                    centroid = centroid.flatten()
                                # Basic check for expected feature dimension (e.g., from OSNet)
                                if centroid.shape[0] == 512: # Common feature dim for OSNet
                                    clusters.append({
                                        'centroid': centroid,
                                        'count': int(c_data['count'])
                                    })
                                else:
                                    logging.warning(f"Skipping cluster for ID {obj_id} due to unexpected feature dimension: {centroid.shape}")
                            else:
                                logging.warning(f"Skipping malformed cluster data for ID {obj_id_str}: {c_data}")
                        if clusters: # Only add if there are valid clusters
                           self.feature_db[obj_id] = clusters
                           valid_ids_loaded.append(obj_id)
                        else:
                            logging.warning(f"No valid clusters loaded for ID {obj_id_str}, skipping this ID.")

                    except (ValueError, TypeError) as e:
                        logging.warning(f"Skipping invalid entry {obj_id_str} in database: {str(e)}")

                if valid_ids_loaded: # Use only successfully loaded and validated IDs
                    self.next_id = max(valid_ids_loaded) + 1 if valid_ids_loaded else 1
                else:
                    self.next_id = 1

                logging.info(f"Loaded database with {len(self.feature_db)} identities. Next ID: {self.next_id}")
            else:
                self.feature_db = {}
                self.next_id = 1
                logging.info("No existing database found. Starting fresh.")

        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from database file: {str(e)}. Initializing empty database.")
            self.feature_db = {}
            self.next_id = 1
        except Exception as e:
            logging.error(f"Critical error loading database: {str(e)}. Initializing empty database.")
            self.feature_db = {} # Fallback to empty DB
            self.next_id = 1

        # Initialize last_active for all loaded IDs
        self.last_active = {k: 0 for k in self.feature_db.keys()}


    def save_database(self):
        try:
            temp_path = self.db_path.with_suffix('.tmp')
            serialized_db = {}
            for k, clusters in self.feature_db.items():
                # Ensure clusters are being saved correctly
                valid_clusters_to_save = []
                for c in clusters:
                    if isinstance(c.get('centroid'), np.ndarray) and isinstance(c.get('count'), int):
                         # Ensure centroid is a flat list of floats
                        centroid_list = c['centroid'].astype(float).tolist()
                        valid_clusters_to_save.append({'centroid': centroid_list, 'count': c['count']})
                    else:
                        logging.warning(f"Skipping saving malformed cluster for ID {k}: {c}")
                if valid_clusters_to_save:
                    serialized_db[str(k)] = valid_clusters_to_save


            with open(temp_path, 'w') as f:
                json.dump(serialized_db, f, indent=2)

            temp_path.replace(self.db_path)
            logging.info(f"Database saved with {len(self.feature_db)} identities")

        except Exception as e:
            logging.error(f"Database save failed: {str(e)}")

    def reset_video_states(self):
        """Reset tracking states that should be cleared for each new video."""
        self.detection_history.clear()
        self.waiting_detections.clear()
        self.previous_detection_count = 0
        self.detection_change_frame = 0
        self.is_in_waiting_period = False
        # Keep feature_db and next_id persistent across videos, but update last_active for pruning
        # self.last_active can be reset or managed globally based on desired behavior
        # For now, let's assume last_active persists globally to prune very old IDs
        logging.info("Per-video tracking states have been reset.")

    def _calculate_iou(self, box1, box2): # Internal helper for clarity
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = box1_area + box2_area - intersection
        return intersection / (union + 1e-9) # Add epsilon for numerical stability

    def _non_max_suppression(self, boxes_data):
        """
        Apply Non-Maximum Suppression.
        Args:
            boxes_data: List of dicts, each with 'bbox', 'conf', 'kpts'.
        Returns:
            List of dicts, filtered by NMS.
        """
        if not boxes_data:
            return []

        # Sort by confidence score in descending order
        boxes_data = sorted(boxes_data, key=lambda x: x['conf'], reverse=True)
        
        keep_indices = []
        # Convert bboxes to a NumPy array for easier slicing if needed, though direct list iteration is fine
        # For simplicity, we'll iterate and mark for removal.
        
        selected_boxes_data = []
        
        while boxes_data:
            chosen_box_data = boxes_data.pop(0) # Get the highest confidence box
            selected_boxes_data.append(chosen_box_data)
            
            # Compare with remaining boxes
            remaining_boxes_data = []
            for box_data in boxes_data:
                iou = self._calculate_iou(chosen_box_data['bbox'], box_data['bbox'])
                if iou < self.NMS_PRE_REID_IOU_THRESHOLD:
                    remaining_boxes_data.append(box_data)
            boxes_data = remaining_boxes_data
            
        return selected_boxes_data

    def check_bbox_size(self, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        return h >= self.MIN_BBOX_SIZE and w >= self.MIN_BBOX_SIZE

    def get_part_boxes(self, crop_shape, keypoints_abs_img):
        """
        Get bounding boxes for body parts based on keypoints.
        Args:
            crop_shape (tuple): (H, W) of the main person crop.
            keypoints_abs_img (np.array): Keypoints with absolute coordinates within the image.
                                         Should be shape (N_kpts, 3) with (x, y, conf).
        Returns:
            list: List of part bounding boxes (px1, py1, px2, py2) relative to the crop.
        """
        h_crop, w_crop = crop_shape
        parts = []
        if keypoints_abs_img is None or keypoints_abs_img.shape[0] == 0:
            return [(0, 0, w_crop, h_crop)] # Return full crop if no keypoints

        # Keypoints are assumed to be in COCO format (17 keypoints)
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
        # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        body_parts_indices = {
            'head': [0, 1, 2, 3, 4],
            'torso': [5, 6, 11, 12], # Shoulders and hips
            'left_arm': [5, 7, 9],
            'right_arm': [6, 8, 10],
            'left_leg': [11, 13, 15],
            'right_leg': [12, 14, 16]
        }

        for part_name, indices in body_parts_indices.items():
            valid_kps_for_part = [] # Relative to the crop
            for idx in indices:
                if idx < len(keypoints_abs_img): # Check if keypoint index is valid
                    kp_x_abs, kp_y_abs, kp_conf = keypoints_abs_img[idx]
                    if kp_conf >= self.PART_CONFIDENCE:
                        # Keypoints are already relative to the crop in batch_extract_features
                        # So, keypoints_abs_img here should actually be keypoints_relative_to_crop
                        valid_kps_for_part.append((kp_x_abs, kp_y_abs))

            if len(valid_kps_for_part) >= 2: # Need at least two points to define a box
                xs = [kp[0] for kp in valid_kps_for_part]
                ys = [kp[1] for kp in valid_kps_for_part]
                
                # Add some padding, ensuring it's within crop boundaries
                padding_w, padding_h = 0.1 * w_crop, 0.1 * h_crop # Padding relative to crop size
                
                px1 = max(0, int(min(xs) - padding_w))
                py1 = max(0, int(min(ys) - padding_h))
                px2 = min(w_crop, int(max(xs) + padding_w))
                py2 = min(h_crop, int(max(ys) + padding_h))

                if px1 < px2 and py1 < py2: # Ensure valid box
                    parts.append((px1, py1, px2, py2))
        
        if not parts: # If no parts found, use the full crop
            return [(0, 0, w_crop, h_crop)]
        return parts

    def batch_extract_features(self, frame, nms_detections_data):
        """
        Extract features from detections.
        Args:
            frame (np.array): The full image frame.
            nms_detections_data (list): List of dicts {'bbox': [x1,y1,x2,y2], 'kpts': keypoints_data}.
                                        Keypoints are absolute to the original frame.
        Returns:
            list: List of tuples (bbox, feature_vector).
        """
        valid_features_info = [] # Will store (bbox_original, aggregated_feature)
        
        # Batch processing lists
        all_crops_to_extract = [] # All part crops (original and flipped)
        person_to_parts_map = defaultdict(list) # Maps original detection index to indices in all_crops_to_extract

        original_bboxes_for_output = [] # Store original bboxes corresponding to features

        for person_idx, det_data in enumerate(nms_detections_data):
            bbox_abs = det_data['bbox']
            keypoints_abs = det_data['kpts'] # Absolute coordinates in the frame

            if not self.check_bbox_size(bbox_abs):
                continue

            x1_abs, y1_abs, x2_abs, y2_abs = map(int, bbox_abs)
            person_crop = frame[y1_abs:y2_abs, x1_abs:x2_abs]

            if person_crop.size == 0 or min(person_crop.shape[:2]) < 10: # Min height/width for a crop
                continue
            
            original_bboxes_for_output.append(bbox_abs) # Keep track of this valid person
            current_person_feature_idx = len(original_bboxes_for_output) - 1


            # Adjust keypoints to be relative to the person_crop
            keypoints_relative_to_crop = None
            if keypoints_abs is not None and keypoints_abs.shape[0] > 0:
                keypoints_relative_to_crop = keypoints_abs.copy()
                keypoints_relative_to_crop[:, 0] -= x1_abs # Adjust x
                keypoints_relative_to_crop[:, 1] -= y1_abs # Adjust y
                # Clip coordinates to be within the crop
                keypoints_relative_to_crop[:, 0] = np.clip(keypoints_relative_to_crop[:, 0], 0, person_crop.shape[1] -1)
                keypoints_relative_to_crop[:, 1] = np.clip(keypoints_relative_to_crop[:, 1], 0, person_crop.shape[0] -1)


            part_boxes_rel = self.get_part_boxes(person_crop.shape[:2], keypoints_relative_to_crop)

            for (px1, py1, px2, py2) in part_boxes_rel:
                part_crop = person_crop[py1:py2, px1:px2]
                if part_crop.size == 0 or min(part_crop.shape[:2]) < 5: # Min height/width for a part
                    continue
                
                all_crops_to_extract.append(part_crop)
                person_to_parts_map[current_person_feature_idx].append(len(all_crops_to_extract) - 1)
                
                all_crops_to_extract.append(cv2.flip(part_crop, 1)) # Horizontal flip augmentation
                person_to_parts_map[current_person_feature_idx].append(len(all_crops_to_extract) - 1)

        if not all_crops_to_extract:
            return []

        try:
            # Batch feature extraction
            extracted_features_tensor = self.extractor(all_crops_to_extract)
            extracted_features_np = extracted_features_tensor.cpu().numpy()
        except Exception as e:
            logging.error(f"Feature extraction failed during batch processing: {str(e)}")
            return []

        # Aggregate features for each person
        for person_idx, part_indices in person_to_parts_map.items():
            if not part_indices:
                continue

            person_features_list = []
            for crop_idx in part_indices:
                feat = extracted_features_np[crop_idx]
                norm_feat = feat / (np.linalg.norm(feat) + 1e-9) # Normalize
                person_features_list.append(norm_feat)
            
            if person_features_list:
                # Average features from all parts (and their flips) for this person
                aggregated_feature = np.mean(person_features_list, axis=0)
                aggregated_feature /= (np.linalg.norm(aggregated_feature) + 1e-9) # Normalize aggregated
                valid_features_info.append((original_bboxes_for_output[person_idx], aggregated_feature))
        
        return valid_features_info


    def match_features(self, query_feature, threshold=None):
        match_threshold = threshold if threshold is not None else self.FEATURE_MATCH_THRESHOLD
        
        if not self.feature_db:
            return None, 0.0 # No database to match against

        query_norm = query_feature / (np.linalg.norm(query_feature) + 1e-9) # Ensure query is normalized

        best_match_id = None
        highest_similarity = -1.0 # Start with a value lower than any possible cosine similarity

        for obj_id, clusters in self.feature_db.items():
            if not clusters: # Skip if an ID somehow has no clusters
                continue
            
            # Calculate similarity with each cluster centroid for this obj_id
            # The ReID model (OSNet) typically produces features where higher dot product means higher similarity
            current_max_sim_for_id = -1.0
            for cluster in clusters:
                centroid_norm = cluster['centroid'] / (np.linalg.norm(cluster['centroid']) + 1e-9)
                similarity = np.dot(query_norm, centroid_norm)
                if similarity > current_max_sim_for_id:
                    current_max_sim_for_id = similarity
            
            if current_max_sim_for_id > highest_similarity:
                highest_similarity = current_max_sim_for_id
                best_match_id = obj_id
        
        if best_match_id is not None and highest_similarity >= match_threshold:
            return best_match_id, highest_similarity
        else:
            # If no match met the threshold, return None for ID, but still return the best similarity found
            return None, highest_similarity 


    def update_feature_store(self, obj_id, new_feature, frame_id):
        new_norm_feature = new_feature / (np.linalg.norm(new_feature) + 1e-9) # Normalize

        if obj_id not in self.feature_db:
            self.feature_db[obj_id] = [{'centroid': new_norm_feature, 'count': 1}]
            self.last_active[obj_id] = frame_id
            return

        clusters = self.feature_db[obj_id]
        
        best_sim_to_cluster = -1.0
        best_cluster_idx = -1

        for i, cluster in enumerate(clusters):
            centroid_norm = cluster['centroid'] / (np.linalg.norm(cluster['centroid']) + 1e-9)
            sim = np.dot(new_norm_feature, centroid_norm)
            if sim > best_sim_to_cluster:
                best_sim_to_cluster = sim
                best_cluster_idx = i
        
        if best_sim_to_cluster >= self.CLUSTER_THRESHOLD and best_cluster_idx != -1:
            # Update existing cluster
            matched_cluster = clusters[best_cluster_idx]
            old_count = matched_cluster['count']
            # Weighted average for the centroid
            matched_cluster['centroid'] = (matched_cluster['centroid'] * old_count + new_norm_feature) / (old_count + 1)
            matched_cluster['centroid'] /= (np.linalg.norm(matched_cluster['centroid']) + 1e-9) # Re-normalize
            matched_cluster['count'] += 1
        else:
            # Add as a new cluster
            clusters.append({'centroid': new_norm_feature, 'count': 1})
            # Prune oldest cluster if exceeding max allowed clusters
            if len(clusters) > self.MAX_CLUSTERS_PER_ID:
                # Sort by count (ascending) to remove least representative, or just oldest (FIFO)
                # clusters.sort(key=lambda c: c['count']) # Option: remove least frequent
                del clusters[0] # Simple FIFO removal of the oldest cluster
        
        self.last_active[obj_id] = frame_id

    def prune_database(self, current_frame_id):
        if not self.feature_db: # No need to prune if DB is empty
            return

        # Prune based on MAX_INACTIVE_FRAMES
        ids_to_prune_inactive = [
            obj_id for obj_id, last_seen_frame in self.last_active.items()
            if current_frame_id - last_seen_frame > self.MAX_INACTIVE_FRAMES
        ]

        pruned_count = 0
        for obj_id in ids_to_prune_inactive:
            if obj_id in self.feature_db:
                del self.feature_db[obj_id]
                pruned_count +=1
            if obj_id in self.last_active:
                del self.last_active[obj_id]
        
        if pruned_count > 0:
            logging.info(f"Pruned {pruned_count} inactive IDs from database.")

    def manage_tracking_states(self, current_detections_count, frame_id):
        # Logic to enter/exit waiting period if number of detected persons changes significantly
        count_changed = current_detections_count != self.previous_detection_count
        
        if count_changed:
            self.is_in_waiting_period = True
            self.detection_change_frame = frame_id
            self.waiting_detections.clear() # Clear previous waiting data
            logging.debug(f"Frame {frame_id}: Detection count changed from {self.previous_detection_count} to {current_detections_count}. Entering waiting period for {self.WAITING_FRAMES} frames.")
        
        self.previous_detection_count = current_detections_count
        return self.is_in_waiting_period # Return current status

    def process_frame(self, frame, frame_id):
        if frame_id % self.PRUNE_INTERVAL == 0 and frame_id > 0:
            self.prune_database(frame_id)
            self.save_database() # Periodically save the database
            
        # 1. Object Detection (YOLO)
        # YOLO results are already somewhat filtered by confidence and NMS internally by YOLO
        yolo_results = self.yolo(frame, classes=0, verbose=False)[0] # class 0 is 'person'
        
        raw_detections_data = [] # To store {'bbox': ..., 'conf': ..., 'kpts': ...}
        for det_idx, det_box in enumerate(yolo_results.boxes.data):
            x1, y1, x2, y2, conf, cls = map(float, det_box)
            if conf < self.DETECTION_CONFIDENCE:
                continue

            kpts_data = None
            if yolo_results.keypoints and det_idx < len(yolo_results.keypoints.data):
                kpts_obj = yolo_results.keypoints[det_idx]
                # Keypoints are usually [x, y, conf] or [x, y, visibility, conf]
                # We expect (N_kpts, 3) with x, y, conf for each keypoint
                kpts_data = kpts_obj.data[0].cpu().numpy() # Shape (num_keypoints, 3) for x,y,conf
            
            raw_detections_data.append({
                'bbox': [x1, y1, x2, y2],
                'conf': conf,
                'kpts': kpts_data
            })

        # 2. Pre-ReID NMS to handle overlapping detections from YOLO itself
        nms_filtered_detections = self._non_max_suppression(raw_detections_data)

        # 3. Feature Extraction for NMS-filtered detections
        # `batch_extract_features` now returns list of (bbox, feature_vector)
        current_frame_features_info = self.batch_extract_features(frame, nms_filtered_detections)
        
        # 4. Manage tracking states (enter/exit waiting period)
        self.manage_tracking_states(len(current_frame_features_info), frame_id)

        # Increment frames_missing for all known tracks
        for track_id in list(self.detection_history.keys()): # Iterate over copy for safe deletion
            if track_id in self.detection_history: # Check if still exists (could be cleaned up)
                 self.detection_history[track_id]['frames_missing'] += 1

        # 5. Cleanup stale tracks (those missing for too long)
        self.cleanup_stale_tracks(frame_id) # Pass frame_id for potential future use in cleanup

        # 6. Assign IDs based on state (initial, waiting, normal)
        output_results = [] # List of (bbox, assigned_id) for this frame

        if not self.feature_db and current_frame_features_info: # DB is empty, first detections
            output_results = self.handle_initial_detections(current_frame_features_info, frame_id)
        elif self.is_in_waiting_period:
            output_results = self.process_waiting_period(current_frame_features_info, frame_id)
        else: # Normal operation
            output_results = self.process_normal_detections(current_frame_features_info, frame_id)
            
        return output_results


    def handle_initial_detections(self, current_features_info, frame_id):
        results = []
        for bbox, features in current_features_info:
            # Bbox size check already done in batch_extract_features, but can double check
            if not self.check_bbox_size(bbox): 
                continue
            
            new_id = self.next_id
            self.next_id += 1
            
            self.update_feature_store(new_id, features, frame_id)
            self.detection_history[new_id] = {
                'bbox': bbox,
                'features': features, # Store the latest representative feature
                'velocity': (0.0, 0.0), # Initial velocity
                'frames_missing': 0,
                'last_seen_frame': frame_id
            }
            results.append((bbox, new_id))
            logging.info(f"Frame {frame_id}: Initial detection, new ID {new_id} assigned.")
        return results

    def process_waiting_period(self, current_features_info, frame_id):
        temp_results_during_wait = [] # (bbox, None) as ID is pending

        # Store all current detections with their features during the waiting period
        for bbox, features in current_features_info:
            if not self.check_bbox_size(bbox):
                continue
            
            # Use a tuple of int bbox coordinates as a key for waiting_detections
            # This is a bit simplistic if bboxes jitter slightly; consider averaging features
            # of spatially close detections if that's an issue.
            # For now, assume each distinct bbox is a potential track.
            detection_key = tuple(map(int, bbox)) 
            self.waiting_detections[detection_key].append({
                'frame_id': frame_id,
                'features': features,
                'bbox': bbox # Store the exact bbox at this frame
            })
            temp_results_during_wait.append((bbox, None)) # ID is None during waiting

        if frame_id - self.detection_change_frame >= self.WAITING_FRAMES:
            self.is_in_waiting_period = False # Exit waiting period
            logging.debug(f"Frame {frame_id}: Exiting waiting period. Confirming tracks.")
            
            confirmed_results = []
            # Consolidate detections from the waiting period
            for det_key_bbox_tuple, history_list in self.waiting_detections.items():
                if len(history_list) >= max(2, int(self.WAITING_FRAMES * 0.3)): # Needs some persistence
                    try:
                        # Use features and bbox from the most recent sighting in history
                        latest_sighting = history_list[-1]
                        # Average features seen for this detection during waiting period
                        avg_features = np.mean([item['features'] for item in history_list], axis=0)
                        avg_features /= (np.linalg.norm(avg_features) + 1e-9)

                        # Try to match this consolidated detection
                        # Use a slightly relaxed feature match threshold after waiting
                        match_id, similarity = self.match_features(avg_features, 
                                                                  threshold=self.FEATURE_MATCH_THRESHOLD * 0.95)

                        if match_id is None: # No existing match, create new ID
                            match_id = self.next_id
                            self.next_id += 1
                            logging.info(f"Frame {frame_id}: Confirmed new ID {match_id} after waiting (sim: {similarity:.2f}).")
                        else:
                            logging.debug(f"Frame {frame_id}: Confirmed match for ID {match_id} after waiting (sim: {similarity:.2f}).")

                        self.update_feature_store(match_id, avg_features, frame_id)
                        self.detection_history[match_id] = {
                            'bbox': latest_sighting['bbox'],
                            'features': avg_features,
                            'velocity': self.detection_history.get(match_id, {}).get('velocity', (0.0,0.0)), # Try to keep old velocity
                            'frames_missing': 0,
                            'last_seen_frame': frame_id
                        }
                        confirmed_results.append((latest_sighting['bbox'], match_id))
                    except Exception as e:
                        logging.error(f"Error processing confirmed detection from waiting period: {str(e)}")
            
            self.waiting_detections.clear() # Clear waiting buffer
            return confirmed_results
        
        return temp_results_during_wait # Still in waiting, return temp results


    def process_normal_detections(self, current_features_info, frame_id):
        assigned_detections_indices = set() # Keep track of assigned current detections
        track_assignments = {} # Map track_id to index of current_features_info

        # 1. Match with existing tracks using IOU and prediction
        unmatched_detection_indices = list(range(len(current_features_info)))
        
        # Create a list of (track_id, predicted_bbox) for active tracks
        active_tracks_predictions = []
        for track_id, track_data in self.detection_history.items():
            if track_data['frames_missing'] == 0: # Consider only recently seen tracks for IOU
                predicted_bbox = self.predict_bbox(track_data['bbox'], track_data.get('velocity', (0,0)))
                active_tracks_predictions.append({'id': track_id, 'bbox': predicted_bbox, 'original_bbox': track_data['bbox']})

        # Cost matrix for Hungarian algorithm or greedy assignment (IOU based)
        # For simplicity, using a greedy approach here.
        # A more robust method would use Hungarian algorithm for optimal assignment.
        
        # Attempt IOU matches first
        potential_iou_matches = [] # (track_id, detection_idx, iou_score)
        temp_unmatched_detection_indices = list(unmatched_detection_indices)

        for track_info in active_tracks_predictions:
            track_id = track_info['id']
            predicted_bbox = track_info['bbox']
            best_iou_for_track = 0
            best_det_idx_for_track = -1

            for det_idx in temp_unmatched_detection_indices:
                bbox_new, _ = current_features_info[det_idx]
                if not self.check_bbox_size(bbox_new): continue

                iou = self._calculate_iou(bbox_new, predicted_bbox)
                if iou > self.TRACK_IOU_THRESHOLD and iou > best_iou_for_track:
                    best_iou_for_track = iou
                    best_det_idx_for_track = det_idx
            
            if best_det_idx_for_track != -1:
                potential_iou_matches.append((track_id, best_det_idx_for_track, best_iou_for_track))

        # Sort potential IOU matches by IOU score (descending) and assign greedily
        potential_iou_matches.sort(key=lambda x: x[2], reverse=True)
        
        final_results = []
        
        for track_id, det_idx, iou_score in potential_iou_matches:
            if det_idx in assigned_detections_indices: continue # Already assigned based on higher IOU

            bbox_new, features_new = current_features_info[det_idx]
            self.update_track(track_id, bbox_new, features_new, frame_id)
            final_results.append((bbox_new, track_id))
            assigned_detections_indices.add(det_idx)
            if det_idx in unmatched_detection_indices: # Should always be true here
                 unmatched_detection_indices.remove(det_idx)
            logging.debug(f"Frame {frame_id}: Matched ID {track_id} by IOU (score: {iou_score:.2f}).")


        # 2. Match remaining detections using features
        temp_unmatched_detection_indices_after_iou = list(unmatched_detection_indices) # Work with a copy

        for det_idx in temp_unmatched_detection_indices_after_iou:
            if det_idx in assigned_detections_indices: continue # Should not happen if logic is correct

            bbox_new, features_new = current_features_info[det_idx]
            if not self.check_bbox_size(bbox_new): continue

            match_id, similarity = self.match_features(features_new)

            if match_id is not None and match_id not in [res[1] for res in final_results if res[1] is not None]: # Ensure ID is not already assigned this frame
                # Check if this matched_id was an IOU match target already.
                # This can happen if one track IOU-matched, and another detection feature-matches the same track.
                # This check helps prevent assigning same track_id to multiple detections in one frame.
                # However, with pre-NMS, this should be less of an issue.
                self.update_track(match_id, bbox_new, features_new, frame_id)
                final_results.append((bbox_new, match_id))
                assigned_detections_indices.add(det_idx)
                unmatched_detection_indices.remove(det_idx)
                logging.debug(f"Frame {frame_id}: Matched ID {match_id} by features (sim: {similarity:.2f}).")
            else:
                # If no feature match or matched ID already used, it might be a new track
                # This will be handled in the next step
                pass
        
        # 3. Handle truly unmatched detections (create new IDs)
        for det_idx in unmatched_detection_indices: # These are definitely unassigned
            if det_idx in assigned_detections_indices: continue # Final safety check

            bbox_new, features_new = current_features_info[det_idx]
            if not self.check_bbox_size(bbox_new): continue

            new_id = self.next_id
            self.next_id += 1
            self.update_track(new_id, bbox_new, features_new, frame_id) # update_track will create if not exists
            final_results.append((bbox_new, new_id))
            logging.info(f"Frame {frame_id}: Created new ID {new_id} for unmatched detection.")
            # No need to add to assigned_detections_indices as we are iterating over remaining unmatched.

        return final_results


    def predict_bbox(self, bbox, velocity):
        # Simple linear prediction
        # bbox: [x1, y1, x2, y2]
        # velocity: (dx_center, dy_center)
        dx, dy = velocity
        return [
            bbox[0] + dx, bbox[1] + dy,
            bbox[2] + dx, bbox[3] + dy
        ]

    def update_track(self, track_id, bbox_new, features_new, frame_id):
        if not self.check_bbox_size(bbox_new):
            # If a track's bbox becomes too small, mark it for faster removal or special handling
            if track_id in self.detection_history:
                self.detection_history[track_id]['frames_missing'] = self.MAX_FRAMES_MISSING + 1 
            return

        prev_bbox = bbox_new # Default if no history
        if track_id in self.detection_history and 'bbox' in self.detection_history[track_id]:
            prev_bbox = self.detection_history[track_id]['bbox']

        # Calculate center points for velocity
        prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
        prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
        new_cx = (bbox_new[0] + bbox_new[2]) / 2
        new_cy = (bbox_new[1] + bbox_new[3]) / 2
        
        # Simple velocity - could be smoothed (e.g., moving average)
        # If track is new, previous velocity doesn't exist or is (0,0)
        # Smoothing factor alpha for EMA
        alpha = 0.5 
        prev_vx, prev_vy = self.detection_history.get(track_id, {}).get('velocity', (0.0, 0.0))

        current_dx = new_cx - prev_cx
        current_dy = new_cy - prev_cy
        
        # Apply Exponential Moving Average for smoother velocity
        vx = alpha * current_dx + (1 - alpha) * prev_vx
        vy = alpha * current_dy + (1 - alpha) * prev_vy
        
        velocity = (vx, vy)

        self.detection_history[track_id] = {
            'bbox': bbox_new,
            'features': features_new, # Store latest representative features
            'velocity': velocity,
            'frames_missing': 0, # Reset missing counter on update
            'last_seen_frame': frame_id
        }
        self.update_feature_store(track_id, features_new, frame_id)
        self.last_active[track_id] = frame_id # Update activity for DB pruning

    def cleanup_stale_tracks(self, frame_id): # Pass frame_id for logging/context
        stale_ids = []
        for track_id, track_data in self.detection_history.items():
            if track_data['frames_missing'] > self.MAX_FRAMES_MISSING:
                stale_ids.append(track_id)
            # Add check for bbox size again, if a track's bbox consistently becomes too small
            elif not self.check_bbox_size(track_data['bbox']) and track_data['frames_missing'] > 0: # Check if it was also missed
                # If bbox is too small AND it has been missed for a few frames, make it stale faster
                if track_data['frames_missing'] > self.MAX_FRAMES_MISSING / 2: # Example heuristic
                     stale_ids.append(track_id)


        for track_id in stale_ids:
            if track_id in self.detection_history:
                del self.detection_history[track_id]
                logging.info(f"Frame {frame_id}: Removed stale track ID {track_id} (missed for {self.MAX_FRAMES_MISSING}+ frames or invalid size).")
            # Note: Feature data in self.feature_db for this ID will be pruned later by prune_database
            # if it remains inactive for self.MAX_INACTIVE_FRAMES.


# Modified main function to process an array of videos
def main(video_paths):
    reid_system = PersonReID(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        # You can adjust other parameters here if needed, e.g.:
        # track_iou_threshold=0.5,
        # feature_match_threshold=0.75,
        # detection_confidence=0.5,
        # nms_pre_reid_iou_threshold=0.4
    )

    global_frame_counter = 0  # Global frame counter across all videos for DB pruning etc.

    for video_idx, video_path in enumerate(video_paths):
        logging.info(f"--- Starting processing of video {video_idx + 1}/{len(video_paths)}: {video_path} ---")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            continue
        
        # Reset per-video states (like detection_history, waiting period)
        # feature_db and next_id are persistent.
        reid_system.reset_video_states() 
        
        video_frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.info(f"Finished processing video: {video_path}")
                break

            # Process the frame with a global frame_id
            results = reid_system.process_frame(frame, global_frame_counter)
            global_frame_counter += 1
            video_frame_num +=1

            # Visualization
            for bbox, obj_id in results:
                x1, y1, x2, y2 = map(int, bbox)
                color = (0, 255, 0) if obj_id is not None else (255, 255, 0) # Cyan for pending
                label = f"ID: {obj_id}" if obj_id is not None else "Detecting..."
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            
            # Display current video path and frame number
            info_text = f"Video: {Path(video_path).name} (Frame: {video_frame_num})"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)


            # Resize for display if needed
            display_frame = cv2.resize(frame, (1280, 720)) # Example resize
            cv2.imshow('Person Re-ID', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("Processing stopped by user (q pressed).")
                cap.release()
                cv2.destroyAllWindows()
                reid_system.save_database() # Save current state before exiting
                logging.info("Database saved. Exiting.")
                return # Exit main function
            elif key == ord('p'): # Pause
                logging.info("Processing paused. Press any key to continue...")
                cv2.waitKey(-1)


        cap.release()
        # Optionally save DB after each video if desired
        # reid_system.save_database() 
    
    cv2.destroyAllWindows()
    reid_system.save_database() # Final save after all videos
    logging.info("--- All videos processed. Database saved. ---")

if __name__ == "__main__":
    # Example list of video paths (replace with your actual video files)
    # Ensure paths are correct and files exist.
    video_files = [
        './videos/20-5-1-4.mp4',  # Make sure these paths are correct relative to where you run the script
        './videos/20-05-4.MOV'
        # Add more video paths here
        # e.g., 'path/to/your/video1.mp4', 'path/to/your/video2.avi'
    ]
    
    # Check if example video paths exist, otherwise use a placeholder
    # This is just for robust example execution, replace with your actual paths.
    valid_video_paths = []
    if not video_files: # If list is empty
        logging.warning("Video paths list is empty. Please provide video paths.")
    else:
        for p in video_files:
            if Path(p).exists():
                valid_video_paths.append(p)
            else:
                logging.warning(f"Video file not found: {p}. Skipping.")

    if not valid_video_paths:
        logging.error("No valid video paths found to process. Please check your video_files list.")
    else:
        main(valid_video_paths)