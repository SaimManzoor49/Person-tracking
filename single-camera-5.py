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
                 waiting_frames=8,
                 min_bbox_size=130,
                 track_iou_threshold=0.99, # Default changed from 0.98 for stricter IOU
                 feature_match_threshold=0.73, # Adjusted for Mahalanobis similarity (0.0 to 1.0)
                 max_frames_missing=120,
                 detection_confidence=0.5,
                 part_confidence=0.6,
                 nms_pre_reid_iou_threshold=0.99):

        self.device = device
        self.yolo = YOLO(yolo_model)
        self.extractor = FeatureExtractor(model_name=reid_model, device=device, model_path='./models/osnet_ain_x1_0_market1501_256x128_amsgrad_ep100_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth')

        self.next_id = 1
        self.detection_history = {}  # Stores current state of tracked IDs
        self.waiting_detections = defaultdict(list) # Stores detections during waiting period
        self.feature_db = {}  # Stores feature clusters for each ID
        self.last_active = {} # Stores last frame_id an ID was active in feature_db
        self.db_path = Path('reid_database.json')

        self.WAITING_FRAMES = waiting_frames
        self.MIN_BBOX_SIZE = min_bbox_size
        self.TRACK_IOU_THRESHOLD = track_iou_threshold
        self.FEATURE_MATCH_THRESHOLD = feature_match_threshold
        self.MAX_FRAMES_MISSING = max_frames_missing
        self.DETECTION_CONFIDENCE = detection_confidence
        self.NMS_PRE_REID_IOU_THRESHOLD = nms_pre_reid_iou_threshold
        self.PART_CONFIDENCE = part_confidence

        # For Mahalanobis sim 1/(1+sqrt(D^2)), cluster threshold should be higher (closer match)
        self.CLUSTER_THRESHOLD = min(self.FEATURE_MATCH_THRESHOLD + 0.1, 0.95)

        self.MAX_CLUSTERS_PER_ID = 10
        self.PRUNE_INTERVAL = 10000 # Frames after which to prune DB
        self.MAX_INACTIVE_FRAMES_DB = 5000 # Max frames an ID can be inactive in feature_db before pruning

        self.previous_detection_count = 0
        self.detection_change_frame = 0
        self.is_in_waiting_period = False

        # --- Settings for new features ---
        self.REID_CROP_SCALES = [1.0, 0.75, 0.5]
        self.MIN_REID_INPUT_DIM = 16

        # --- Feature Smoothing ---
        self.FEATURE_SMOOTHING_ALPHA = 0.3 # For EMA of feature vectors

        # --- Stale Track Recovery ---
        self.stale_track_cache = {} # {track_id: data}
        self.MAX_STALE_TRACK_CACHE_SIZE = 50
        self.MAX_STALE_TRACK_AGE_FRAMES = 300 # Approx 10s at 30fps
        # For Mahalanobis, a lower threshold is more lenient.
        self.STALE_TRACK_RECOVERY_THRESHOLD = max(0.1, self.FEATURE_MATCH_THRESHOLD - 0.1)


        # --- Mahalanobis Distance Settings ---
        try:
            dummy_img = np.zeros((128, 64, 3), dtype=np.uint8)
            dummy_feat = self.extractor([dummy_img])
            self.feature_dim = dummy_feat.shape[1]
            logging.info(f"Determined feature dimension: {self.feature_dim}")
        except Exception as e:
            logging.warning(f"Could not dynamically determine feature dimension, defaulting to 512. Error: {e}")
            self.feature_dim = 512

        self.global_covariance_matrix = np.eye(self.feature_dim)
        self.global_inv_covariance_matrix = np.eye(self.feature_dim)
        self.feature_buffer_for_cov = deque(maxlen=max(2000, 2 * self.feature_dim))
        self.cov_update_interval = 500
        self.min_features_for_cov_estimation = max(100, self.feature_dim + 10) # Ensure enough samples

        logging.warning("Mahalanobis distance is used. 'feature_match_threshold' (currently %.2f) and "
                        "'CLUSTER_THRESHOLD' (currently %.2f) apply to Mahalanobis similarity (1/(1+sqrt(D^2))) "
                        "and may need tuning.", self.FEATURE_MATCH_THRESHOLD, self.CLUSTER_THRESHOLD)
        logging.info(f"Stale track recovery threshold: {self.STALE_TRACK_RECOVERY_THRESHOLD:.2f}")

        self.load_database()

    def _update_global_covariance_matrix(self):
        if len(self.feature_buffer_for_cov) < self.min_features_for_cov_estimation:
            logging.debug(f"Covariance: Not enough features ({len(self.feature_buffer_for_cov)}/{self.min_features_for_cov_estimation}). Using identity.")
            self.global_covariance_matrix = np.eye(self.feature_dim)
            self.global_inv_covariance_matrix = np.eye(self.feature_dim)
            return

        features_np = np.array(list(self.feature_buffer_for_cov))
        cov_matrix = np.cov(features_np, rowvar=False)
        reg_lambda = 1e-5
        self.global_covariance_matrix = cov_matrix + reg_lambda * np.eye(self.feature_dim)
        
        try:
            self.global_inv_covariance_matrix = np.linalg.inv(self.global_covariance_matrix)
            logging.info(f"Global covariance matrix updated using {len(self.feature_buffer_for_cov)} features.")
        except np.linalg.LinAlgError:
            logging.warning("Covariance matrix inversion failed. Using pseudo-inverse.")
            self.global_inv_covariance_matrix = np.linalg.pinv(self.global_covariance_matrix)
        

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
                                if centroid.ndim > 1: centroid = centroid.flatten()
                                if centroid.shape[0] == self.feature_dim:
                                    clusters.append({'centroid': centroid, 'count': int(c_data['count'])})
                                else:
                                    logging.warning(f"DB Load: Skip cluster ID {obj_id}, bad dim: {centroid.shape}, expected {self.feature_dim}")
                            else:
                                logging.warning(f"DB Load: Skip malformed cluster ID {obj_id_str}: {c_data}")
                        if clusters:
                           self.feature_db[obj_id] = clusters
                           valid_ids_loaded.append(obj_id)
                           # No need to populate feature_buffer_for_cov from DB, it's for runtime features
                        else:
                            logging.warning(f"DB Load: No valid clusters for ID {obj_id_str}.")
                    except (ValueError, TypeError) as e:
                        logging.warning(f"DB Load: Skipping invalid entry {obj_id_str}: {str(e)}")

                self.next_id = max(valid_ids_loaded) + 1 if valid_ids_loaded else 1
                logging.info(f"Loaded database with {len(self.feature_db)} IDs. Next ID: {self.next_id}")
            else:
                self.feature_db = {}
                self.next_id = 1
                logging.info("No existing database found. Starting fresh.")
        except Exception as e: # Catch broader exceptions
            logging.error(f"Critical error loading database: {str(e)}. Initializing empty database.")
            self.feature_db = {}
            self.next_id = 1
        self.last_active = {k: 0 for k in self.feature_db.keys()} # Initialize last_active for DB pruning

    def save_database(self):
        try:
            temp_path = self.db_path.with_suffix('.tmp')
            serialized_db = {}
            for k, clusters in self.feature_db.items():
                valid_clusters_to_save = []
                for c in clusters:
                    if isinstance(c.get('centroid'), np.ndarray) and isinstance(c.get('count'), int):
                        centroid_list = c['centroid'].astype(float).tolist()
                        valid_clusters_to_save.append({'centroid': centroid_list, 'count': c['count']})
                    else:
                        logging.warning(f"DB Save: Skipping malformed cluster for ID {k}: {c}")
                if valid_clusters_to_save:
                    serialized_db[str(k)] = valid_clusters_to_save
            with open(temp_path, 'w') as f:
                json.dump(serialized_db, f, indent=2)
            temp_path.replace(self.db_path)
            logging.info(f"Database saved with {len(self.feature_db)} identities")
        except Exception as e:
            logging.error(f"Database save failed: {str(e)}")

    def reset_video_states(self):
        self.detection_history.clear()
        self.waiting_detections.clear()
        self.stale_track_cache.clear() # Also clear stale cache for new video
        self.previous_detection_count = 0
        self.detection_change_frame = 0
        self.is_in_waiting_period = False
        logging.info("Per-video tracking states have been reset.")

    def _calculate_iou(self, box1, box2):
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        return intersection / (union + 1e-9)

    def _non_max_suppression(self, boxes_data):
        if not boxes_data: return []
        boxes_data = sorted(boxes_data, key=lambda x: x['conf'], reverse=True)
        selected_boxes_data = []
        while boxes_data:
            chosen = boxes_data.pop(0)
            selected_boxes_data.append(chosen)
            boxes_data = [b for b in boxes_data if self._calculate_iou(chosen['bbox'], b['bbox']) < self.NMS_PRE_REID_IOU_THRESHOLD]
        return selected_boxes_data

    def check_bbox_size(self, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        return h >= self.MIN_BBOX_SIZE and w >= self.MIN_BBOX_SIZE

    def get_part_boxes(self, crop_shape, keypoints_rel_crop):
        h_crop, w_crop = crop_shape
        parts = []
        if keypoints_rel_crop is None or keypoints_rel_crop.shape[0] == 0: return []
        body_parts_indices = {
            'torso': [5, 6, 11, 12], 'left_leg': [11, 13, 15], 'right_leg': [12, 14, 16]
        }
        for _, indices in body_parts_indices.items():
            valid_kps = [(k[0], k[1]) for i in indices if i < len(keypoints_rel_crop) and (k := keypoints_rel_crop[i])[2] >= self.PART_CONFIDENCE]
            if len(valid_kps) >= 2:
                xs, ys = [kp[0] for kp in valid_kps], [kp[1] for kp in valid_kps]
                pad_w, pad_h = int(0.1 * w_crop), int(0.1 * h_crop)
                px1, py1 = max(0, int(min(xs)) - pad_w), max(0, int(min(ys)) - pad_h)
                px2, py2 = min(w_crop, int(max(xs)) + pad_w), min(h_crop, int(max(ys)) + pad_h)
                if px1 < px2 and py1 < py2: parts.append((px1, py1, px2, py2))
        return parts

    def _add_crop_to_list(self, crop_image, target_list, person_map_entry_list):
        if crop_image.size > 0 and crop_image.shape[0] >= self.MIN_REID_INPUT_DIM and crop_image.shape[1] >= self.MIN_REID_INPUT_DIM:
            target_list.append(crop_image)
            person_map_entry_list.append(len(target_list) - 1)

    def batch_extract_features(self, frame, nms_detections_data):
        valid_features_info = []
        all_crops_to_extract, person_to_crop_indices_map, original_bboxes_for_output = [], defaultdict(list), []

        for det_data in nms_detections_data:
            bbox_abs, keypoints_abs = det_data['bbox'], det_data['kpts']
            if not self.check_bbox_size(bbox_abs): continue
            x1_abs, y1_abs, x2_abs, y2_abs = map(int, bbox_abs)
            person_crop = frame[y1_abs:y2_abs, x1_abs:x2_abs]
            if person_crop.size == 0 or min(person_crop.shape[:2]) < self.MIN_REID_INPUT_DIM: continue
            
            current_person_map_key = len(original_bboxes_for_output)
            original_bboxes_for_output.append(bbox_abs)
            map_list_entry = person_to_crop_indices_map[current_person_map_key]

            for scale in self.REID_CROP_SCALES:
                scaled_crop = person_crop if scale == 1.0 else cv2.resize(person_crop, (int(person_crop.shape[1]*scale), int(person_crop.shape[0]*scale)))
                if scaled_crop.shape[0] < self.MIN_REID_INPUT_DIM or scaled_crop.shape[1] < self.MIN_REID_INPUT_DIM: continue
                self._add_crop_to_list(scaled_crop, all_crops_to_extract, map_list_entry)
                self._add_crop_to_list(cv2.flip(scaled_crop, 1), all_crops_to_extract, map_list_entry)

            kpts_rel = None
            if keypoints_abs is not None and keypoints_abs.shape[0] > 0:
                kpts_rel = keypoints_abs.copy()
                kpts_rel[:, 0] = np.clip(kpts_rel[:, 0] - x1_abs, 0, person_crop.shape[1] - 1)
                kpts_rel[:, 1] = np.clip(kpts_rel[:, 1] - y1_abs, 0, person_crop.shape[0] - 1)
            
            for px1, py1, px2, py2 in self.get_part_boxes(person_crop.shape[:2], kpts_rel):
                part_img = person_crop[py1:py2, px1:px2]
                self._add_crop_to_list(part_img, all_crops_to_extract, map_list_entry)
                self._add_crop_to_list(cv2.flip(part_img, 1), all_crops_to_extract, map_list_entry)
            
            if not map_list_entry: # No valid crops for this person
                original_bboxes_for_output.pop()
                del person_to_crop_indices_map[current_person_map_key]

        if not all_crops_to_extract: return []
        try:
            features_tensor = self.extractor(all_crops_to_extract)
            features_np = features_tensor.cpu().numpy()
        except Exception as e:
            logging.error(f"Batch feature extraction failed: {str(e)}")
            return []

        for person_key, crop_indices in person_to_crop_indices_map.items():
            if not crop_indices: continue
            person_feats = [features_np[i] / (np.linalg.norm(features_np[i]) + 1e-9) for i in crop_indices]
            if person_feats:
                agg_feat = np.mean(person_feats, axis=0)
                agg_feat /= (np.linalg.norm(agg_feat) + 1e-9)
                self.feature_buffer_for_cov.append(agg_feat) # Add raw aggregated feature for covariance
                valid_features_info.append((original_bboxes_for_output[person_key], agg_feat))
        return valid_features_info

    def _calculate_mahalanobis_similarity(self, feat1, feat2):
        diff = feat1 - feat2
        dist_sq = diff.T @ self.global_inv_covariance_matrix @ diff
        if dist_sq < 0: dist_sq = 0.0 # Ensure non-negative
        return 1.0 / (1.0 + np.sqrt(dist_sq))

    def match_features(self, query_feature, threshold=None):
        # query_feature is raw aggregated feature for current detection
        if not self.feature_db: return None, 0.0
        best_match_id, highest_avg_similarity = None, -1.0

        for obj_id, clusters in self.feature_db.items():
            if not clusters: continue
            total_weighted_sim, total_weight = 0.0, 0.0
            for cluster in clusters:
                # Cluster centroids are built from SMOOTHED features
                sim = self._calculate_mahalanobis_similarity(query_feature, cluster['centroid'])
                weight = cluster['count']
                total_weighted_sim += sim * weight
                total_weight += weight
            if total_weight > 0:
                avg_sim = total_weighted_sim / total_weight
                if avg_sim > highest_avg_similarity:
                    highest_avg_similarity, best_match_id = avg_sim, obj_id
        
        eff_thresh = threshold if threshold is not None else self.FEATURE_MATCH_THRESHOLD
        return (best_match_id, highest_avg_similarity) if best_match_id and highest_avg_similarity >= eff_thresh else (None, highest_avg_similarity)

    def update_feature_store(self, obj_id, smoothed_feature_to_store, frame_id):
        # smoothed_feature_to_store is the temporally smoothed feature for the track
        if obj_id not in self.feature_db:
            self.feature_db[obj_id] = [{'centroid': smoothed_feature_to_store, 'count': 1}]
            self.last_active[obj_id] = frame_id
            return

        clusters = self.feature_db[obj_id]
        best_sim_to_cluster, best_cluster_idx = -1.0, -1
        for i, cluster in enumerate(clusters):
            sim = self._calculate_mahalanobis_similarity(smoothed_feature_to_store, cluster['centroid'])
            if sim > best_sim_to_cluster:
                best_sim_to_cluster, best_cluster_idx = sim, i
        
        if best_sim_to_cluster >= self.CLUSTER_THRESHOLD and best_cluster_idx != -1:
            matched_cluster = clusters[best_cluster_idx]
            old_count = matched_cluster['count']
            matched_cluster['centroid'] = (matched_cluster['centroid'] * old_count + smoothed_feature_to_store) / (old_count + 1)
            matched_cluster['centroid'] /= (np.linalg.norm(matched_cluster['centroid']) + 1e-9)
            matched_cluster['count'] += 1
        else:
            clusters.append({'centroid': smoothed_feature_to_store, 'count': 1})
            if len(clusters) > self.MAX_CLUSTERS_PER_ID: del clusters[0]
        self.last_active[obj_id] = frame_id

    def prune_database(self, current_frame_id):
        if not self.feature_db: return
        ids_to_prune = [oid for oid, last_f in self.last_active.items() if current_frame_id - last_f > self.MAX_INACTIVE_FRAMES_DB]
        pruned_count = 0
        for obj_id in ids_to_prune:
            if obj_id in self.feature_db: del self.feature_db[obj_id]; pruned_count +=1
            if obj_id in self.last_active: del self.last_active[obj_id]
        if pruned_count > 0: logging.info(f"Pruned {pruned_count} inactive IDs from feature_db.")

    def manage_tracking_states(self, current_detections_count, frame_id):
        if current_detections_count != self.previous_detection_count:
            self.is_in_waiting_period = True
            self.detection_change_frame = frame_id
            self.waiting_detections.clear()
            logging.debug(f"Frame {frame_id}: Detection count changed. Entering waiting period.")
        self.previous_detection_count = current_detections_count
        return self.is_in_waiting_period

    def process_frame(self, frame, frame_id):
        if frame_id > 0 and frame_id % self.PRUNE_INTERVAL == 0:
            self.prune_database(frame_id)
            self.save_database()
        if frame_id > 0 and frame_id % self.cov_update_interval == 0:
            self._update_global_covariance_matrix()

        yolo_results = self.yolo(frame, classes=0, verbose=False)[0]
        raw_detections_data = []
        for det_idx, det_box in enumerate(yolo_results.boxes.data):
            x1, y1, x2, y2, conf, _ = map(float, det_box)
            if conf < self.DETECTION_CONFIDENCE: continue
            kpts = yolo_results.keypoints[det_idx].data[0].cpu().numpy() if yolo_results.keypoints and det_idx < len(yolo_results.keypoints.data) else None
            raw_detections_data.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'kpts': kpts})

        nms_detections = self._non_max_suppression(raw_detections_data)
        current_features_info = self.batch_extract_features(frame, nms_detections) # List of (bbox, raw_aggregated_feature)
        
        self.manage_tracking_states(len(current_features_info), frame_id)

        for track_id in list(self.detection_history.keys()): # Increment missing frames for active tracks
            self.detection_history[track_id]['frames_missing'] += 1

        self.cleanup_stale_tracks(frame_id) # Moves to stale_track_cache or removes very old stale tracks

        if not self.feature_db and not self.stale_track_cache and current_features_info: # DB and stale cache are empty
            return self.handle_initial_detections(current_features_info, frame_id)
        elif self.is_in_waiting_period:
            return self.process_waiting_period(current_features_info, frame_id)
        else:
            return self.process_normal_detections(current_features_info, frame_id)

    def handle_initial_detections(self, current_features_info, frame_id):
        results = []
        for bbox, raw_feature in current_features_info: # raw_feature is current frame's aggregated feature
            if not self.check_bbox_size(bbox): continue
            new_id = self.next_id; self.next_id += 1
            # Initial smoothed feature is just the raw feature
            self.update_feature_store(new_id, raw_feature, frame_id) # Store initial (effectively smoothed) feature
            self.detection_history[new_id] = {
                'bbox': bbox, 'smoothed_feature': raw_feature, 'velocity': (0.0, 0.0),
                'frames_missing': 0, 'last_seen_frame': frame_id
            }
            results.append((bbox, new_id))
            logging.info(f"Frame {frame_id}: Initial detection, new ID {new_id} assigned.")
        return results

    def process_waiting_period(self, current_features_info, frame_id):
        temp_results = []
        for bbox, raw_feature in current_features_info:
            if not self.check_bbox_size(bbox): continue
            self.waiting_detections[tuple(map(int, bbox))].append({
                'frame_id': frame_id, 'raw_feature': raw_feature, 'bbox': bbox
            })
            temp_results.append((bbox, None))

        if frame_id - self.detection_change_frame >= self.WAITING_FRAMES:
            self.is_in_waiting_period = False
            logging.debug(f"Frame {frame_id}: Exiting waiting period. Confirming tracks.")
            confirmed_results = []
            for _, history in self.waiting_detections.items():
                if len(history) >= max(2, int(self.WAITING_FRAMES * 0.3)):
                    latest = history[-1]
                    # Averaged raw features from waiting period becomes the initial smoothed feature
                    avg_raw_feature = np.mean([item['raw_feature'] for item in history], axis=0)
                    avg_raw_feature /= (np.linalg.norm(avg_raw_feature) + 1e-9)
                    
                    # Match against existing DB. For Mahalanobis, a lower threshold is more lenient.
                    # FEATURE_MATCH_THRESHOLD is e.g. 0.6. Relaxed could be 0.5.
                    relaxed_thresh = max(0.1, self.FEATURE_MATCH_THRESHOLD - 0.1) 
                    match_id, sim = self.match_features(avg_raw_feature, threshold=relaxed_thresh)

                    if match_id is None: # Try to match against stale cache before creating new ID
                        match_id, sim = self._match_against_stale_cache(avg_raw_feature, threshold=self.STALE_TRACK_RECOVERY_THRESHOLD)
                        if match_id is not None: # Recovered from stale
                             logging.info(f"Frame {frame_id}: Recovered stale ID {match_id} after waiting (sim: {sim:.2f}).")
                             # Recovery logic will be handled by update_track if called
                        else: # Truly new ID
                            match_id = self.next_id; self.next_id += 1
                            logging.info(f"Frame {frame_id}: Confirmed new ID {match_id} after waiting (sim to DB: {sim:.2f}).")
                    else: # Matched existing ID in DB
                        logging.debug(f"Frame {frame_id}: Confirmed match for ID {match_id} after waiting (sim: {sim:.2f}).")
                    
                    self.update_track(match_id, latest['bbox'], avg_raw_feature, frame_id, is_recovery=(match_id in self.stale_track_cache))
                    confirmed_results.append((latest['bbox'], match_id))
            self.waiting_detections.clear()
            return confirmed_results
        return temp_results

    def _match_against_stale_cache(self, query_raw_feature, threshold):
        if not self.stale_track_cache: return None, 0.0
        best_stale_id, highest_sim = None, -1.0
        for stale_id, stale_data in self.stale_track_cache.items():
            # Compare current raw feature against stale track's smoothed feature
            sim = self._calculate_mahalanobis_similarity(query_raw_feature, stale_data['smoothed_feature'])
            if sim > highest_sim:
                highest_sim, best_stale_id = sim, stale_id
        return (best_stale_id, highest_sim) if best_stale_id and highest_sim >= threshold else (None, highest_sim)

    def process_normal_detections(self, current_features_info, frame_id):
        assigned_det_indices, final_results = set(), []
        unmatched_det_indices = list(range(len(current_features_info)))
        
        # 1. IOU Matching
        active_preds = [{'id': tid, 'bbox': self.predict_bbox(tdata['bbox'], tdata.get('velocity',(0,0)))} 
                        for tid, tdata in self.detection_history.items() if tdata['frames_missing'] == 0]
        iou_matches = []
        for track_info in active_preds:
            best_iou, best_idx = 0, -1
            for det_idx in unmatched_det_indices:
                if det_idx in assigned_det_indices: continue
                bbox_new, _ = current_features_info[det_idx]
                if not self.check_bbox_size(bbox_new): continue
                iou = self._calculate_iou(bbox_new, track_info['bbox'])
                if iou > self.TRACK_IOU_THRESHOLD and iou > best_iou: best_iou, best_idx = iou, det_idx
            if best_idx != -1: iou_matches.append((track_info['id'], best_idx, best_iou))
        
        iou_matches.sort(key=lambda x: x[2], reverse=True)
        for track_id, det_idx, iou in iou_matches:
            if det_idx in assigned_det_indices: continue
            bbox_new, raw_feat = current_features_info[det_idx]
            self.update_track(track_id, bbox_new, raw_feat, frame_id)
            final_results.append((bbox_new, track_id)); assigned_det_indices.add(det_idx)
            logging.debug(f"Frame {frame_id}: Matched ID {track_id} by IOU (score: {iou:.2f}).")
        
        unmatched_det_indices = [i for i in unmatched_det_indices if i not in assigned_det_indices]

        # 2. Feature Matching (against feature_db)
        feature_matches_found_ids = set() # To avoid assigning same DB ID to multiple new detections
        for det_idx in list(unmatched_det_indices): # Iterate copy for modification
            if det_idx in assigned_det_indices: continue
            bbox_new, raw_feat = current_features_info[det_idx]
            if not self.check_bbox_size(bbox_new): continue
            
            match_id, sim = self.match_features(raw_feat) # Compares raw_feat against feature_db clusters
            if match_id is not None and match_id not in feature_matches_found_ids:
                # Ensure this matched ID is not already an active track from IOU match
                if match_id in self.detection_history and self.detection_history[match_id]['frames_missing'] == 0 and any(res[1] == match_id for res in final_results):
                    logging.debug(f"Frame {frame_id}: Feature match for ID {match_id} (sim: {sim:.2f}) but ID already assigned by IOU. Skipping.")
                    continue # ID already used by IOU match this frame

                self.update_track(match_id, bbox_new, raw_feat, frame_id)
                final_results.append((bbox_new, match_id)); assigned_det_indices.add(det_idx)
                feature_matches_found_ids.add(match_id)
                unmatched_det_indices.remove(det_idx)
                logging.debug(f"Frame {frame_id}: Matched ID {match_id} by features (sim: {sim:.2f}).")
        
        # 3. Stale Track Recovery
        for det_idx in list(unmatched_det_indices): # Iterate copy for modification
            if det_idx in assigned_det_indices: continue
            bbox_new, raw_feat = current_features_info[det_idx]
            if not self.check_bbox_size(bbox_new): continue

            recovered_id, sim = self._match_against_stale_cache(raw_feat, threshold=self.STALE_TRACK_RECOVERY_THRESHOLD)
            if recovered_id is not None:
                self.update_track(recovered_id, bbox_new, raw_feat, frame_id, is_recovery=True)
                final_results.append((bbox_new, recovered_id)); assigned_det_indices.add(det_idx)
                unmatched_det_indices.remove(det_idx)
                logging.info(f"Frame {frame_id}: Recovered stale ID {recovered_id} (sim: {sim:.2f}).")
        
        # 4. New ID Assignment
        for det_idx in unmatched_det_indices:
            if det_idx in assigned_det_indices: continue # Should not happen
            bbox_new, raw_feat = current_features_info[det_idx]
            if not self.check_bbox_size(bbox_new): continue
            new_id = self.next_id; self.next_id += 1
            self.update_track(new_id, bbox_new, raw_feat, frame_id) # Initializes with raw_feat as first smoothed_feat
            final_results.append((bbox_new, new_id))
            logging.info(f"Frame {frame_id}: Created new ID {new_id} for unmatched detection.")
            
        return final_results

    def predict_bbox(self, bbox, velocity):
        dx, dy = velocity
        return [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]

    def update_track(self, track_id, bbox_new, current_frame_raw_feature, frame_id, is_recovery=False):
        if not self.check_bbox_size(bbox_new):
            if track_id in self.detection_history:
                self.detection_history[track_id]['frames_missing'] = self.MAX_FRAMES_MISSING + 1 
            return

        # Feature Smoothing
        if is_recovery and track_id in self.stale_track_cache: # Recovering a stale track
            # Use the stale track's last known smoothed feature
            prev_smoothed_feature = self.stale_track_cache[track_id]['smoothed_feature']
            # Move from stale cache to active history
            self.detection_history[track_id] = self.stale_track_cache.pop(track_id) 
            self.detection_history[track_id].pop('stale_since_frame', None) # remove stale marker
            self.detection_history[track_id].pop('last_seen_frame_active', None) # remove stale marker
        elif track_id in self.detection_history: # Existing active track
            prev_smoothed_feature = self.detection_history[track_id]['smoothed_feature']
        else: # New track
            prev_smoothed_feature = current_frame_raw_feature # Initial smoothed feature is the first raw feature

        # EMA for smoothed feature
        new_smoothed_feature = (1 - self.FEATURE_SMOOTHING_ALPHA) * prev_smoothed_feature + \
                               self.FEATURE_SMOOTHING_ALPHA * current_frame_raw_feature
        new_smoothed_feature /= (np.linalg.norm(new_smoothed_feature) + 1e-9)

        # Velocity calculation
        prev_bbox_for_velo = self.detection_history.get(track_id, {}).get('bbox', bbox_new)
        prev_cx = (prev_bbox_for_velo[0] + prev_bbox_for_velo[2]) / 2
        prev_cy = (prev_bbox_for_velo[1] + prev_bbox_for_velo[3]) / 2
        new_cx = (bbox_new[0] + bbox_new[2]) / 2
        new_cy = (bbox_new[1] + bbox_new[3]) / 2
        
        alpha_v = 0.5 
        prev_vx, prev_vy = self.detection_history.get(track_id, {}).get('velocity', (0.0, 0.0))
        dx, dy = new_cx - prev_cx, new_cy - prev_cy
        vx = alpha_v * dx + (1 - alpha_v) * prev_vx
        vy = alpha_v * dy + (1 - alpha_v) * prev_vy
        
        self.detection_history[track_id] = {
            'bbox': bbox_new, 'smoothed_feature': new_smoothed_feature, 'velocity': (vx, vy),
            'frames_missing': 0, 'last_seen_frame': frame_id
        }
        self.update_feature_store(track_id, new_smoothed_feature, frame_id) # Store smoothed feature in DB
        self.last_active[track_id] = frame_id # For DB pruning

    def cleanup_stale_tracks(self, frame_id):
        # Move tracks from detection_history to stale_track_cache
        stale_ids_to_move = []
        for track_id, track_data in self.detection_history.items():
            if track_data['frames_missing'] > self.MAX_FRAMES_MISSING or \
               (not self.check_bbox_size(track_data['bbox']) and track_data['frames_missing'] > self.MAX_FRAMES_MISSING / 2):
                stale_ids_to_move.append(track_id)

        for track_id in stale_ids_to_move:
            if track_id in self.detection_history:
                if len(self.stale_track_cache) >= self.MAX_STALE_TRACK_CACHE_SIZE:
                    # Remove oldest from stale_track_cache if full
                    oldest_stale_id = min(self.stale_track_cache, key=lambda k: self.stale_track_cache[k]['stale_since_frame'], default=None)
                    if oldest_stale_id: del self.stale_track_cache[oldest_stale_id]
                
                track_to_stale = self.detection_history.pop(track_id)
                self.stale_track_cache[track_id] = {
                    'smoothed_feature': track_to_stale['smoothed_feature'],
                    'bbox': track_to_stale['bbox'], # Last known bbox
                    'velocity': track_to_stale.get('velocity', (0,0)), # Last known velocity
                    'last_seen_frame_active': track_to_stale['last_seen_frame'],
                    'stale_since_frame': frame_id
                }
                logging.info(f"Frame {frame_id}: Moved track ID {track_id} to stale cache.")

        # Remove very old tracks from stale_track_cache
        ids_to_purge_from_stale = [
            stale_id for stale_id, data in self.stale_track_cache.items()
            if frame_id - data['stale_since_frame'] > self.MAX_STALE_TRACK_AGE_FRAMES
        ]
        for stale_id in ids_to_purge_from_stale:
            del self.stale_track_cache[stale_id]
            logging.info(f"Frame {frame_id}: Purged very old track ID {stale_id} from stale cache.")


def main(video_paths):
    reid_system = PersonReID(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        feature_match_threshold=0.6, # Example for Mahalanobis, may need tuning
        track_iou_threshold=0.5, # Looser IOU for more associations before feature matching
        max_frames_missing=60 # Shorter time before track becomes stale
    )
    global_frame_counter = 0 
    for video_idx, video_path in enumerate(video_paths):
        logging.info(f"--- Processing video {video_idx + 1}/{len(video_paths)}: {video_path} ---")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): logging.error(f"Failed to open: {video_path}"); continue
        reid_system.reset_video_states() 
        video_frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret: logging.info(f"Finished video: {video_path}"); break
            results = reid_system.process_frame(frame, global_frame_counter)
            global_frame_counter += 1; video_frame_num +=1
            for bbox, obj_id in results:
                x1, y1, x2, y2 = map(int, bbox)
                color = (0, 255, 0) if obj_id is not None else (0, 255, 255) # Yellow for pending
                label = f"ID: {obj_id}" if obj_id is not None else "Wait..."
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            info = f"{Path(video_path).name} (F:{video_frame_num}) Act:{len(reid_system.detection_history)} Stale:{len(reid_system.stale_track_cache)}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow('Person Re-ID', cv2.resize(frame, (1280, 720)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): cap.release(); cv2.destroyAllWindows(); reid_system.save_database(); logging.info("Exit."); return
            elif key == ord('p'): logging.info("Paused."); cv2.waitKey(-1)
        cap.release()
    cv2.destroyAllWindows()
    reid_system.save_database()
    logging.info("--- All videos processed. ---")

if __name__ == "__main__":
    video_files = [
        './videos/20-5-1-4.mp4',
        './videos/20-05-4.MOV'
    ]
    valid_paths = [p for p in video_files if Path(p).exists()]
    if not video_files: logging.warning("Video list empty.")
    elif not valid_paths: logging.error("No valid video paths found.")
    else: main(valid_paths)