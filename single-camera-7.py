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
                 device='cpu',
                 waiting_frames=12, # Duration for a new detection to be "pending"
                 min_bbox_size=125,
                 track_iou_threshold=0.7, # Adjusted for more general tracking before ReID confirms
                 feature_match_threshold=0.64,
                 max_frames_missing=120, # For active tracks to become stale
                 detection_confidence=0.5,
                 part_confidence=0.5,
                 nms_pre_reid_iou_threshold=0.7):

        self.device = device
        self.yolo = YOLO(yolo_model)
        self.extractor = FeatureExtractor(model_name=reid_model, device=device, model_path='./models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth')

        self.next_id = 1
        self.detection_history = {} # Stores active tracks: {track_id: data}
        
        self.pending_tracks = {} # Stores tracks awaiting confirmation: {temp_id: data}
        self.next_pending_id_counter = 0
        
        self.feature_db = {}
        self.last_active = {} # For DB pruning based on last_seen frame for an ID
        self.db_path = Path('reid_database.json')

        self.WAITING_FRAMES = waiting_frames 
        self.MIN_BBOX_SIZE = min_bbox_size
        self.TRACK_IOU_THRESHOLD = track_iou_threshold
        self.FEATURE_MATCH_THRESHOLD = feature_match_threshold
        self.MAX_FRAMES_MISSING = max_frames_missing # For active tracks
        self.DETECTION_CONFIDENCE = detection_confidence
        self.NMS_PRE_REID_IOU_THRESHOLD = nms_pre_reid_iou_threshold
        self.PART_CONFIDENCE = part_confidence

        self.CLUSTER_THRESHOLD = min(self.FEATURE_MATCH_THRESHOLD + 0.03, 0.85)
        self.MAX_CLUSTERS_PER_ID = 10
        self.PRUNE_INTERVAL = 10000
        self.MAX_INACTIVE_FRAMES_DB = 5000

        self.REID_CROP_SCALES = [1.0, 0.75, 0.5]
        self.MIN_REID_INPUT_DIM = 16
        self.FEATURE_SMOOTHING_ALPHA = 0.3

        self.stale_track_cache = {}
        self.MAX_STALE_TRACK_CACHE_SIZE = 50
        self.MAX_STALE_TRACK_AGE_FRAMES = 300
        self.STALE_TRACK_RECOVERY_THRESHOLD = max(0.1, self.FEATURE_MATCH_THRESHOLD - 0.1)
        
        # For pending tracks specifically
        self.PENDING_IOU_THRESHOLD = 0.2 # Lower IOU to associate current detection with a pending track
        self.MAX_MISS_FOR_PENDING_TRACK = int(self.WAITING_FRAMES / 2) # If a pending track isn't seen for this long

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
        self.min_features_for_cov_estimation = max(100, self.feature_dim + 10)

        self.similarity_log_successful_match = deque(maxlen=1000)
        self.similarity_log_interval = 600
        
        logging.warning("Mahalanobis distance is used. 'feature_match_threshold' (currently %.2f) and "
                        "'CLUSTER_THRESHOLD' (currently %.2f) apply to Mahalanobis similarity (1/(1+sqrt(D^2))) "
                        "and may need tuning.", self.FEATURE_MATCH_THRESHOLD, self.CLUSTER_THRESHOLD)
        logging.info(f"Stale track recovery threshold: {self.STALE_TRACK_RECOVERY_THRESHOLD:.2f}")
        logging.info(f"MIN_BBOX_SIZE set to: {self.MIN_BBOX_SIZE} pixels for H and W.")
        logging.info(f"WAITING_FRAMES for new detections set to: {self.WAITING_FRAMES}")


        self.load_database()

    def _log_similarity_stats(self, frame_id):
        if frame_id > 0 and frame_id % self.similarity_log_interval == 0:
            if self.similarity_log_successful_match:
                scores = np.array(list(self.similarity_log_successful_match))
                logging.info(f"Frame {frame_id} - Successful Match Similarity Stats (last {len(scores)} matches): "
                             f"Mean={np.mean(scores):.3f}, Median={np.median(scores):.3f}, "
                             f"Std={np.std(scores):.3f}, Min={np.min(scores):.3f}, Max={np.max(scores):.3f}")
            else:
                logging.info(f"Frame {frame_id} - No successful matches recorded recently to log similarity stats.")


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
        except Exception as e:
            logging.error(f"Critical error loading database: {str(e)}. Initializing empty database.")
            self.feature_db = {}
            self.next_id = 1
        self.last_active = {k: 0 for k in self.feature_db.keys()} # Initialize last_active for all loaded DB IDs

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
        self.pending_tracks.clear()
        self.next_pending_id_counter = 0
        self.stale_track_cache.clear()
        self.similarity_log_successful_match.clear()
        # self.last_active is intentionally not cleared here, as it pertains to the global DB
        logging.info("Per-video tracking states (active, pending, stale) have been reset.")

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
        h = y2 - y1
        w = x2 - x1
        return h >= self.MIN_BBOX_SIZE and w >= self.MIN_BBOX_SIZE

    def get_part_boxes(self, crop_shape, keypoints_rel_crop):
        h_crop, w_crop = crop_shape
        parts = []
        if keypoints_rel_crop is None or not isinstance(keypoints_rel_crop, np.ndarray) or keypoints_rel_crop.shape[0] == 0: 
            return []
        body_parts_indices = {
            'torso': [5, 6, 11, 12], 'left_leg': [11, 13, 15], 'right_leg': [12, 14, 16]
        }
        for _, indices in body_parts_indices.items():
            valid_kps = []
            if keypoints_rel_crop.ndim == 2 and keypoints_rel_crop.shape[1] == 3: 
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

        for det_data_idx, det_data in enumerate(nms_detections_data):
            bbox_abs, keypoints_abs = det_data['bbox'], det_data['kpts']
            
            x1_temp, y1_temp, x2_temp, y2_temp = map(int, bbox_abs)
            h_temp, w_temp = y2_temp - y1_temp, x2_temp - x1_temp
            if not self.check_bbox_size(bbox_abs): # Initial check on raw detection
                logging.debug(f"BatchExtract: BBox [{x1_temp},{y1_temp},{x2_temp},{y2_temp}] (H:{h_temp}, W:{w_temp}) "
                              f"for det_idx {det_data_idx} failed MIN_BBOX_SIZE ({self.MIN_BBOX_SIZE}). Skipping feature extraction.")
                continue
            
            person_crop = frame[y1_temp:y2_temp, x1_temp:x2_temp]
            if person_crop.size == 0 or min(person_crop.shape[:2]) < self.MIN_REID_INPUT_DIM:
                logging.debug(f"BatchExtract: Person crop for det_idx {det_data_idx} too small or empty. Skipping.")
                continue
            
            current_person_map_key = len(original_bboxes_for_output)
            original_bboxes_for_output.append(bbox_abs)
            map_list_entry = person_to_crop_indices_map[current_person_map_key]

            for scale in self.REID_CROP_SCALES:
                target_w, target_h = int(person_crop.shape[1]*scale), int(person_crop.shape[0]*scale)
                if target_w < self.MIN_REID_INPUT_DIM or target_h < self.MIN_REID_INPUT_DIM : continue
                scaled_crop = person_crop if scale == 1.0 else cv2.resize(person_crop, (target_w, target_h))
                self._add_crop_to_list(scaled_crop, all_crops_to_extract, map_list_entry)
                self._add_crop_to_list(cv2.flip(scaled_crop, 1), all_crops_to_extract, map_list_entry)

            kpts_rel = None
            if keypoints_abs is not None and isinstance(keypoints_abs, np.ndarray) and keypoints_abs.shape[0] > 0:
                kpts_rel = keypoints_abs.copy()
                kpts_rel[:, 0] = np.clip(kpts_rel[:, 0] - x1_temp, 0, person_crop.shape[1] - 1)
                kpts_rel[:, 1] = np.clip(kpts_rel[:, 1] - y1_temp, 0, person_crop.shape[0] - 1)
            
            for px1, py1, px2, py2 in self.get_part_boxes(person_crop.shape[:2], kpts_rel):
                part_img = person_crop[py1:py2, px1:px2]
                self._add_crop_to_list(part_img, all_crops_to_extract, map_list_entry)
                self._add_crop_to_list(cv2.flip(part_img, 1), all_crops_to_extract, map_list_entry)
            
            if not map_list_entry: 
                if current_person_map_key in person_to_crop_indices_map: # Should always be true if we got here
                    del person_to_crop_indices_map[current_person_map_key]
                # We also need to remove the bbox from original_bboxes_for_output.
                # This is handled by the reconstruction logic below.


        if not all_crops_to_extract: return []
        try:
            features_tensor = self.extractor(all_crops_to_extract)
            features_np = features_tensor.cpu().numpy()
        except Exception as e:
            logging.error(f"Batch feature extraction failed: {str(e)}")
            return []
        
        retained_original_bboxes = []
        final_person_to_crop_map = defaultdict(list)
        new_person_key_idx = 0

        sorted_original_keys = sorted(person_to_crop_indices_map.keys())

        for original_key in sorted_original_keys: # original_key is the index in the initial original_bboxes_for_output
            crop_indices_for_person = person_to_crop_indices_map[original_key]
            if crop_indices_for_person: 
                retained_original_bboxes.append(original_bboxes_for_output[original_key])
                final_person_to_crop_map[new_person_key_idx] = crop_indices_for_person
                new_person_key_idx +=1
        
        original_bboxes_for_output = retained_original_bboxes 
        person_to_crop_indices_map = final_person_to_crop_map


        for person_key, crop_indices in person_to_crop_indices_map.items(): # person_key is now an index for the filtered list
            if not crop_indices: continue 

            person_feats = [features_np[i] / (np.linalg.norm(features_np[i]) + 1e-9) for i in crop_indices]
            if person_feats:
                agg_feat = np.mean(person_feats, axis=0)
                agg_feat /= (np.linalg.norm(agg_feat) + 1e-9)
                self.feature_buffer_for_cov.append(agg_feat.copy())
                valid_features_info.append((original_bboxes_for_output[person_key], agg_feat))
        return valid_features_info

    def _calculate_mahalanobis_similarity(self, feat1, feat2):
        diff = feat1 - feat2
        dist_sq = diff.T @ self.global_inv_covariance_matrix @ diff
        if dist_sq < 0: dist_sq = 0.0 
        return 1.0 / (1.0 + np.sqrt(dist_sq))

    def match_features(self, query_feature, threshold=None):
        if not self.feature_db: return None, 0.0
        best_match_id, highest_avg_similarity = None, -1.0

        for obj_id, clusters in self.feature_db.items():
            if not clusters: continue
            
            total_weighted_sim, total_weight = 0.0, 0.0
            for cluster in clusters:
                sim = self._calculate_mahalanobis_similarity(query_feature, cluster['centroid'])
                weight = cluster['count'] 
                total_weighted_sim += sim * weight
                total_weight += weight
            if total_weight > 0:
                avg_sim = total_weighted_sim / total_weight
                if avg_sim > highest_avg_similarity:
                    highest_avg_similarity, best_match_id = avg_sim, obj_id
        
        eff_thresh = threshold if threshold is not None else self.FEATURE_MATCH_THRESHOLD
        if best_match_id and highest_avg_similarity >= eff_thresh:
            self.similarity_log_successful_match.append(highest_avg_similarity)
            return best_match_id, highest_avg_similarity
        return None, highest_avg_similarity


    def update_feature_store(self, obj_id, smoothed_feature_to_store, frame_id):
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
            if len(clusters) > self.MAX_CLUSTERS_PER_ID: 
                clusters.sort(key=lambda c: c['count']) 
                del clusters[0] 
        self.last_active[obj_id] = frame_id


    def prune_database(self, current_frame_id):
        if not self.feature_db: return
        ids_to_prune = [oid for oid, last_f in self.last_active.items() if current_frame_id - last_f > self.MAX_INACTIVE_FRAMES_DB]
        pruned_count = 0
        for obj_id in ids_to_prune:
            if obj_id in self.feature_db: del self.feature_db[obj_id]; pruned_count +=1
            if obj_id in self.last_active: del self.last_active[obj_id]
        if pruned_count > 0: logging.info(f"Pruned {pruned_count} inactive IDs from feature_db.")


    def process_frame(self, frame, frame_id):
        self._log_similarity_stats(frame_id)

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
            
            kpts = None
            if yolo_results.keypoints is not None and hasattr(yolo_results.keypoints, 'data') and \
               yolo_results.keypoints.data.nelement() > 0 and \
               det_idx < yolo_results.keypoints.data.shape[0]:
                kpts_tensor_for_det = yolo_results.keypoints.data[det_idx]
                if kpts_tensor_for_det.nelement() > 0:
                     kpts = kpts_tensor_for_det.cpu().numpy()
            
            raw_detections_data.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'kpts': kpts})

        nms_detections = self._non_max_suppression(raw_detections_data)
        current_features_info = self.batch_extract_features(frame, nms_detections) 
        
        final_results_this_frame = []
        assigned_det_indices = set() 

        for track_id in list(self.detection_history.keys()):
            self.detection_history[track_id]['frames_missing'] += 1

        active_tracks_for_iou = []
        for tid, tdata in self.detection_history.items():
            if tdata['frames_missing'] < self.MAX_FRAMES_MISSING / 4 : 
                 active_tracks_for_iou.append({'id': tid, 'bbox': self.predict_bbox(tdata['bbox'], tdata.get('velocity',(0,0)))})
        
        iou_matches = [] 
        for track_info in active_tracks_for_iou:
            best_iou_for_track, best_det_idx_for_track = 0, -1
            for det_idx, (bbox_new, _) in enumerate(current_features_info):
                if det_idx in assigned_det_indices: continue
                iou_val = self._calculate_iou(bbox_new, track_info['bbox'])
                if iou_val > self.TRACK_IOU_THRESHOLD and iou_val > best_iou_for_track:
                    best_iou_for_track = iou_val
                    best_det_idx_for_track = det_idx
            if best_det_idx_for_track != -1:
                iou_matches.append((track_info['id'], best_det_idx_for_track, best_iou_for_track))
        
        iou_matches.sort(key=lambda x: x[2], reverse=True) 
        for track_id, det_idx, iou_score in iou_matches:
            if det_idx in assigned_det_indices: continue 
            is_track_id_already_used = any(res[1] == track_id for res in final_results_this_frame)
            if is_track_id_already_used: continue

            bbox_new, raw_feat = current_features_info[det_idx]
            self.update_track(track_id, bbox_new, raw_feat, frame_id)
            final_results_this_frame.append((bbox_new, track_id))
            assigned_det_indices.add(det_idx)
            logging.debug(f"Frame {frame_id}: Matched ID {track_id} to det_idx {det_idx} by IOU (score: {iou_score:.2f}).")

        feature_matched_ids_this_frame = set(res[1] for res in final_results_this_frame if res[1] is not None and not isinstance(res[1], str))
        
        for det_idx, (bbox_new, raw_feat) in enumerate(current_features_info):
            if det_idx in assigned_det_indices: continue

            match_id, sim = self.match_features(raw_feat)
            if match_id is not None and match_id not in feature_matched_ids_this_frame:
                self.update_track(match_id, bbox_new, raw_feat, frame_id)
                final_results_this_frame.append((bbox_new, match_id))
                assigned_det_indices.add(det_idx)
                feature_matched_ids_this_frame.add(match_id) 
                logging.debug(f"Frame {frame_id}: Matched ID {match_id} to det_idx {det_idx} by features (sim: {sim:.2f}).")
        
        for det_idx, (bbox_new, raw_feat) in enumerate(current_features_info):
            if det_idx in assigned_det_indices: continue

            recovered_id, sim_stale = self._match_against_stale_cache(raw_feat, threshold=self.STALE_TRACK_RECOVERY_THRESHOLD)
            if recovered_id is not None and recovered_id not in feature_matched_ids_this_frame:
                self.update_track(recovered_id, bbox_new, raw_feat, frame_id, is_recovery=True)
                final_results_this_frame.append((bbox_new, recovered_id))
                assigned_det_indices.add(det_idx)
                feature_matched_ids_this_frame.add(recovered_id)
                logging.info(f"Frame {frame_id}: Recovered stale ID {recovered_id} for det_idx {det_idx} (sim: {sim_stale:.2f}).")

        next_pending_tracks = {}
        pending_tracks_to_remove_this_step = set()

        for temp_id, p_data in list(self.pending_tracks.items()): 
            p_bbox_current = p_data['display_bbox']
            p_feature_history = p_data['feature_history']
            # p_bbox_history = p_data['bbox_history'] # Not strictly used in current logic flow below but present in data structure
            p_first_seen = p_data['first_seen_frame']
            # p_last_seen = p_data['last_seen_frame'] # Updated if seen

            found_match_for_pending = False
            best_iou_for_pending, matched_det_idx_for_pending = 0, -1

            for det_idx, (bbox_curr_det, _) in enumerate(current_features_info):
                if det_idx in assigned_det_indices: continue 
                iou = self._calculate_iou(p_bbox_current, bbox_curr_det)
                if iou > self.PENDING_IOU_THRESHOLD and iou > best_iou_for_pending:
                    best_iou_for_pending = iou
                    matched_det_idx_for_pending = det_idx
            
            if matched_det_idx_for_pending != -1: 
                found_match_for_pending = True
                assigned_det_indices.add(matched_det_idx_for_pending) 
                
                current_bbox_of_pending, current_raw_feat_of_pending = current_features_info[matched_det_idx_for_pending]
                
                p_feature_history.append(current_raw_feat_of_pending)
                if 'bbox_history' in p_data: p_data['bbox_history'].append(current_bbox_of_pending) # Check if exists
                p_data['display_bbox'] = current_bbox_of_pending
                p_data['last_seen_frame'] = frame_id
                p_data['missed_frames_count'] = 0 # Reset missed count
                
                avg_feat = np.mean(list(p_feature_history), axis=0)
                avg_feat /= (np.linalg.norm(avg_feat) + 1e-9)

                confirmed_id, sim = self.match_features(avg_feat)
                is_stale_recovery = False
                if confirmed_id is None:
                    confirmed_id, sim = self._match_against_stale_cache(avg_feat, threshold=self.STALE_TRACK_RECOVERY_THRESHOLD)
                    if confirmed_id is not None: is_stale_recovery = True
                
                if confirmed_id is not None and confirmed_id not in feature_matched_ids_this_frame:
                    self.update_track(confirmed_id, current_bbox_of_pending, avg_feat, frame_id, is_recovery=is_stale_recovery)
                    final_results_this_frame.append((current_bbox_of_pending, confirmed_id))
                    feature_matched_ids_this_frame.add(confirmed_id)
                    pending_tracks_to_remove_this_step.add(temp_id)
                    logging.info(f"Frame {frame_id}: Confirmed pending track {temp_id} as {'stale ' if is_stale_recovery else ''}ID {confirmed_id} (sim: {sim:.2f}).")
                else: 
                    final_results_this_frame.append((current_bbox_of_pending, temp_id))
                    next_pending_tracks[temp_id] = p_data 
            else: 
                p_data['missed_frames_count'] = p_data.get('missed_frames_count', 0) + 1
                if p_data['missed_frames_count'] > self.MAX_MISS_FOR_PENDING_TRACK:
                    pending_tracks_to_remove_this_step.add(temp_id)
                    logging.debug(f"Frame {frame_id}: Pending track {temp_id} discarded due to missing too many frames ({p_data['missed_frames_count']}).")
                    continue 

            if temp_id not in pending_tracks_to_remove_this_step and (frame_id - p_first_seen >= self.WAITING_FRAMES):
                avg_feat = np.mean(list(p_feature_history), axis=0)
                avg_feat /= (np.linalg.norm(avg_feat) + 1e-9)
                
                final_id, sim = self.match_features(avg_feat) 
                is_stale_recovery = False
                if final_id is None:
                    final_id, sim = self._match_against_stale_cache(avg_feat, threshold=self.STALE_TRACK_RECOVERY_THRESHOLD) 
                    if final_id is not None: is_stale_recovery = True

                current_display_bbox_for_finalized_pending = p_data['display_bbox']

                if final_id is not None and final_id not in feature_matched_ids_this_frame:
                    self.update_track(final_id, current_display_bbox_for_finalized_pending, avg_feat, frame_id, is_recovery=is_stale_recovery)
                    updated_in_results = False
                    for i, res_tuple in enumerate(final_results_this_frame):
                        if res_tuple[1] == temp_id: 
                            final_results_this_frame[i] = (res_tuple[0], final_id)
                            updated_in_results = True
                            break
                    if not updated_in_results:
                         final_results_this_frame.append((current_display_bbox_for_finalized_pending, final_id))
                    feature_matched_ids_this_frame.add(final_id)
                    logging.info(f"Frame {frame_id}: Pending track {temp_id} expired, confirmed as {'stale ' if is_stale_recovery else ''}ID {final_id} (sim: {sim:.2f}).")
                else: 
                    new_actual_id = self.next_id; self.next_id += 1
                    self.update_track(new_actual_id, current_display_bbox_for_finalized_pending, avg_feat, frame_id)
                    updated_in_results = False
                    for i, res_tuple in enumerate(final_results_this_frame):
                        if res_tuple[1] == temp_id: 
                            final_results_this_frame[i] = (res_tuple[0], new_actual_id)
                            updated_in_results = True
                            break
                    if not updated_in_results:
                         final_results_this_frame.append((current_display_bbox_for_finalized_pending, new_actual_id))
                    feature_matched_ids_this_frame.add(new_actual_id)
                    logging.info(f"Frame {frame_id}: Pending track {temp_id} expired, assigned new ID {new_actual_id}.")
                pending_tracks_to_remove_this_step.add(temp_id)
            
            elif not found_match_for_pending and temp_id not in pending_tracks_to_remove_this_step:
                if not any(res[1] == temp_id for res in final_results_this_frame): # Avoid duplicate display if already added
                    final_results_this_frame.append((p_data['display_bbox'], temp_id)) 
                next_pending_tracks[temp_id] = p_data

        self.pending_tracks = {k:v for k,v in next_pending_tracks.items() if k not in pending_tracks_to_remove_this_step}


        for det_idx, (bbox, raw_feat) in enumerate(current_features_info):
            if det_idx in assigned_det_indices: continue

            new_temp_id = f"pending_{self.next_pending_id_counter}"
            self.next_pending_id_counter += 1
            
            self.pending_tracks[new_temp_id] = {
                'display_bbox': bbox,
                'feature_history': deque([raw_feat], maxlen=self.WAITING_FRAMES + 5), 
                'bbox_history': deque([bbox], maxlen=self.WAITING_FRAMES + 5), # Ensure this is present
                'first_seen_frame': frame_id,
                'last_seen_frame': frame_id,
                'missed_frames_count': 0
            }
            final_results_this_frame.append((bbox, new_temp_id))
            # assigned_det_indices.add(det_idx) # Not strictly necessary as loop ends
            logging.info(f"Frame {frame_id}: New detection (det_idx {det_idx}) became pending track {new_temp_id}.")

        self.cleanup_stale_tracks(frame_id)
        
        # --- ADDED: Final Filter for BBox Size before returning results for display ---
        filtered_display_results = []
        for bbox_res, id_res in final_results_this_frame:
            if self.check_bbox_size(bbox_res):
                filtered_display_results.append((bbox_res, id_res))
            else:
                # Optionally log that a track is being suppressed from display
                logging.debug(f"Frame {frame_id}: Suppressing display of ID/Pending {id_res} due to small bbox: {bbox_res} (H:{int(bbox_res[3]-bbox_res[1])}, W:{int(bbox_res[2]-bbox_res[0])})")
        
        return filtered_display_results


    def _match_against_stale_cache(self, query_raw_feature, threshold):
        if not self.stale_track_cache: return None, 0.0
        best_stale_id, highest_sim = None, -1.0
        for stale_id, stale_data in self.stale_track_cache.items():
            sim = self._calculate_mahalanobis_similarity(query_raw_feature, stale_data['smoothed_feature'])
            if sim > highest_sim:
                highest_sim, best_stale_id = sim, stale_id
        
        if best_stale_id and highest_sim >= threshold:
            self.similarity_log_successful_match.append(highest_sim)
            return best_stale_id, highest_sim
        return None, highest_sim


    def predict_bbox(self, bbox, velocity):
        dx, dy = velocity
        return [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]

    def update_track(self, track_id, bbox_new, feature_to_incorporate, frame_id, is_recovery=False):
        if is_recovery and track_id in self.stale_track_cache:
            prev_smoothed_feature = self.stale_track_cache[track_id]['smoothed_feature']
            recovered_data = self.stale_track_cache.pop(track_id) 
            self.detection_history[track_id] = {
                'bbox': recovered_data.get('bbox', bbox_new),
                'smoothed_feature': prev_smoothed_feature, 
                'velocity': recovered_data.get('velocity', (0.0,0.0)),
                'frames_missing': 0, 
                'last_seen_frame': recovered_data.get('last_seen_frame_active', frame_id)
            }
            logging.debug(f"Frame {frame_id}: Recovered ID {track_id} from stale cache into active tracking.")
        elif track_id in self.detection_history: 
            prev_smoothed_feature = self.detection_history[track_id]['smoothed_feature']
        else: 
            prev_smoothed_feature = feature_to_incorporate 
            self.detection_history[track_id] = {'velocity': (0.0,0.0)} 
            logging.debug(f"Frame {frame_id}: Initializing new active track for ID {track_id}.")


        new_smoothed_feature = (1 - self.FEATURE_SMOOTHING_ALPHA) * prev_smoothed_feature + \
                               self.FEATURE_SMOOTHING_ALPHA * feature_to_incorporate
        new_smoothed_feature /= (np.linalg.norm(new_smoothed_feature) + 1e-9)

        prev_bbox_for_velo = self.detection_history.get(track_id, {}).get('bbox', bbox_new) 
        prev_cx = (prev_bbox_for_velo[0] + prev_bbox_for_velo[2]) / 2
        prev_cy = (prev_bbox_for_velo[1] + prev_bbox_for_velo[3]) / 2
        new_cx = (bbox_new[0] + bbox_new[2]) / 2
        new_cy = (bbox_new[1] + bbox_new[3]) / 2
        
        alpha_v = 0.5 
        prev_vx, prev_vy = self.detection_history.get(track_id, {}).get('velocity', (0.0, 0.0))
        dx, dy = 0,0
        if 'bbox' in self.detection_history.get(track_id, {}): 
            dx, dy = new_cx - prev_cx, new_cy - prev_cy
        
        vx = alpha_v * dx + (1 - alpha_v) * prev_vx
        vy = alpha_v * dy + (1 - alpha_v) * prev_vy
        
        self.detection_history[track_id].update({
            'bbox': bbox_new, 
            'smoothed_feature': new_smoothed_feature, 
            'velocity': (vx, vy),
            'frames_missing': 0, 
            'last_seen_frame': frame_id
        })
        self.update_feature_store(track_id, new_smoothed_feature, frame_id) 
        self.last_active[track_id] = frame_id 

    def cleanup_stale_tracks(self, frame_id): 
        stale_ids_to_move = []
        for track_id, track_data in list(self.detection_history.items()): # Iterate copy
            bbox_valid = isinstance(track_data.get('bbox'), (list, np.ndarray)) and len(track_data['bbox']) == 4
            is_too_small_and_missed = False
            if bbox_valid:
                 is_too_small_and_missed = not self.check_bbox_size(track_data['bbox']) and \
                                           track_data['frames_missing'] > self.MAX_FRAMES_MISSING / 2 
            elif track_data['frames_missing'] > self.MAX_FRAMES_MISSING / 2 : 
                    is_too_small_and_missed = True 

            if track_data['frames_missing'] > self.MAX_FRAMES_MISSING or is_too_small_and_missed:
                stale_ids_to_move.append(track_id)

        for track_id in stale_ids_to_move:
            if track_id in self.detection_history:
                if len(self.stale_track_cache) >= self.MAX_STALE_TRACK_CACHE_SIZE:
                    oldest_stale_id = min(self.stale_track_cache, 
                                          key=lambda k: self.stale_track_cache[k]['stale_since_frame'], 
                                          default=None)
                    if oldest_stale_id and oldest_stale_id in self.stale_track_cache:
                        del self.stale_track_cache[oldest_stale_id]
                
                track_to_stale = self.detection_history.pop(track_id)
                if 'smoothed_feature' in track_to_stale and 'bbox' in track_to_stale:
                    self.stale_track_cache[track_id] = {
                        'smoothed_feature': track_to_stale['smoothed_feature'],
                        'bbox': track_to_stale['bbox'],
                        'velocity': track_to_stale.get('velocity', (0,0)),
                        'last_seen_frame_active': track_to_stale['last_seen_frame'],
                        'stale_since_frame': frame_id
                    }
                    logging.info(f"Frame {frame_id}: Moved active track ID {track_id} to stale cache.")
                else:
                    logging.warning(f"Frame {frame_id}: Tried to move ID {track_id} to stale, but essential data missing.")


        ids_to_purge_from_stale = [
            stale_id for stale_id, data in self.stale_track_cache.items()
            if frame_id - data['stale_since_frame'] > self.MAX_STALE_TRACK_AGE_FRAMES
        ]
        for stale_id in ids_to_purge_from_stale:
            if stale_id in self.stale_track_cache:
                del self.stale_track_cache[stale_id]
                logging.info(f"Frame {frame_id}: Purged very old track ID {stale_id} from stale cache.")


def main(video_paths):
    reid_system = PersonReID(
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    global_frame_counter = 0 
    for video_idx, video_path_str in enumerate(video_paths):
        video_path = Path(video_path_str)
        if not video_path.exists():
            logging.error(f"Video path {video_path} does not exist. Skipping.")
            continue

        logging.info(f"--- Processing video {video_idx + 1}/{len(video_paths)}: {video_path.name} ---")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): 
            logging.error(f"Failed to open: {video_path.name}")
            continue
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target_display_width = 1280
        target_display_height = int(target_display_width * (frame_height / frame_width)) if frame_width > 0 and frame_height > 0 else 720

        reid_system.reset_video_states() 
        video_frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret: 
                logging.info(f"Finished video: {video_path.name}")
                break
            
            processed_results = reid_system.process_frame(frame.copy(), global_frame_counter)
            
            global_frame_counter += 1
            video_frame_num +=1
            for bbox, obj_id in processed_results: # This now uses the filtered list
                x1, y1, x2, y2 = map(int, bbox)
                color = (0, 255, 0) 
                label_prefix = "ID: "
                if isinstance(obj_id, str) and obj_id.startswith("pending_"):
                    color = (0, 255, 255) 
                    label_prefix = "PEND: " 
                    obj_id_display = obj_id.split('_')[-1] 
                else:
                    obj_id_display = str(obj_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label_prefix}{obj_id_display}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            info_active = f"Active: {len(reid_system.detection_history)}"
            info_pending = f"Pending: {len(reid_system.pending_tracks)}"
            info_stale = f"Stale: {len(reid_system.stale_track_cache)}"
            info_db = f"DB IDs: {len(reid_system.feature_db)}"
            info_next_id = f"NextID: {reid_system.next_id}"
            
            cv2.putText(frame, f"{video_path.name} (F:{video_frame_num})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, info_active, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)
            cv2.putText(frame, info_pending, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,255), 2) 
            cv2.putText(frame, info_stale, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
            cv2.putText(frame, info_db, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,200), 2)
            cv2.putText(frame, info_next_id, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,200), 2)


            if target_display_width > 0 and target_display_height > 0:
                display_frame = cv2.resize(frame, (target_display_width, target_display_height))
            else:
                display_frame = frame

            cv2.imshow('Person Re-ID', display_frame)
            key = cv2.waitKey(1) & 0xFF 
            if key == ord('q'): 
                cap.release()
                cv2.destroyAllWindows()
                reid_system.save_database()
                logging.info("Exit.")
                return
            elif key == ord('p'): 
                logging.info("Paused. Press any key in OpenCV window to continue...")
                cv2.waitKey(-1)
        cap.release()
    cv2.destroyAllWindows()
    reid_system.save_database()
    logging.info("--- All videos processed. ---")

if __name__ == "__main__":
    video_files = [
        './videos/20-5-1-4.mp4', 
        # './videos/your_other_video.mp4',
    ]
    valid_paths = [p for p in video_files if Path(p).exists() and Path(p).is_file()]
    if not video_files: 
        logging.warning("Video list empty.")
    elif not valid_paths and video_files: 
        logging.error(f"Video files specified but not found: {video_files}. Check paths.")
    elif not valid_paths: 
        logging.error("No valid video paths found to process.")
    else: 
        if len(valid_paths) < len(video_files):
            logging.warning(f"Some specified videos not found. Processing: {valid_paths}")
        main(valid_paths)