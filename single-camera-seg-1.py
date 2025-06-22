import json
import logging
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from collections import defaultdict, deque
import cv2

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s') # Changed to DEBUG for more info

class PersonReID:
    def __init__(self,
                 yolo_model='yolov8m-seg.onnx',
                 reid_model='osnet_ain_x1_0',
                 device='cpu',
                 waiting_frames=22,
                 min_bbox_size=665,
                 track_iou_threshold=0.4,
                 feature_match_threshold=0.70, #46
                 max_frames_missing=580,
                 detection_confidence=0.95,
                 nms_pre_reid_iou_threshold=0.4):

        self.device = device
        self.yolo = YOLO(yolo_model)
        logging.info(f"YOLO model {yolo_model} loaded.")
        self.extractor = FeatureExtractor(model_name=reid_model, device=device, model_path='./models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth')
        logging.info(f"ReID model {reid_model} loaded.")

        self.next_id = 1
        self.detection_history = {}
        self.pending_tracks = {}
        self.next_pending_id_counter = 0
        
        self.feature_db = {}
        self.last_active = {}
        self.db_path = Path('reid_database_seg.json')

        self.WAITING_FRAMES = waiting_frames 
        self.MIN_BBOX_SIZE = min_bbox_size
        self.TRACK_IOU_THRESHOLD = track_iou_threshold
        self.FEATURE_MATCH_THRESHOLD = feature_match_threshold
        self.MAX_FRAMES_MISSING = max_frames_missing
        self.DETECTION_CONFIDENCE = detection_confidence
        self.NMS_PRE_REID_IOU_THRESHOLD = nms_pre_reid_iou_threshold

        self.CLUSTER_THRESHOLD = min(self.FEATURE_MATCH_THRESHOLD + 0.06, 0.85)
        self.MAX_CLUSTERS_PER_ID = 10
        self.PRUNE_INTERVAL = 10000
        self.MAX_INACTIVE_FRAMES_DB = 5000

        self.MIN_REID_INPUT_DIM = 16 
        self.FEATURE_SMOOTHING_ALPHA = 0.3

        self.stale_track_cache = {}
        self.MAX_STALE_TRACK_CACHE_SIZE = 50
        self.MAX_STALE_TRACK_AGE_FRAMES = 300
        self.STALE_TRACK_RECOVERY_THRESHOLD = max(0.1, self.FEATURE_MATCH_THRESHOLD - 0.1)
        
        self.PENDING_IOU_THRESHOLD = 0.2
        self.MAX_MISS_FOR_PENDING_TRACK = int(self.WAITING_FRAMES / 2)

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
        logging.info(f"MIN_BBOX_SIZE set to: {self.MIN_BBOX_SIZE} pixels for H and W (for detection box and display box).")
        logging.info(f"WAITING_FRAMES for new detections set to: {self.WAITING_FRAMES}")
        logging.info(f"YOLO model set to: {yolo_model} (segmentation-based)")
        logging.info(f"DETECTION_CONFIDENCE: {self.DETECTION_CONFIDENCE}")

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
        self.last_active = {k: 0 for k in self.feature_db.keys()}

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

    def _add_crop_to_list(self, crop_image, target_list, person_map_entry_list):
        if crop_image.size > 0 and \
           crop_image.shape[0] >= self.MIN_REID_INPUT_DIM and \
           crop_image.shape[1] >= self.MIN_REID_INPUT_DIM:
            target_list.append(crop_image)
            person_map_entry_list.append(len(target_list) - 1)
            return True
        return False

    def batch_extract_features(self, frame, nms_detections_data):
        valid_features_info = [] 
        all_crops_to_extract = []
        person_to_crop_indices_map = defaultdict(list) 
        original_bboxes_and_masks_for_output = [] 

        logging.debug(f"BatchExtract: Received {len(nms_detections_data)} detections after NMS.")
        frame_h, frame_w = frame.shape[:2]

        for det_idx, det_data in enumerate(nms_detections_data):
            detection_bbox_abs = det_data['bbox'] 
            mask_poly_points = det_data.get('mask_polygon') 

            logging.debug(f"  BatchExtract Det {det_idx}: Input BBox={detection_bbox_abs}, Mask points shape: {mask_poly_points.shape if mask_poly_points is not None else 'None'}")

            if not self.check_bbox_size(detection_bbox_abs):
                h_det, w_det = detection_bbox_abs[3]-detection_bbox_abs[1], detection_bbox_abs[2]-detection_bbox_abs[0]
                logging.debug(f"    BatchExtract Det {det_idx}: Initial BBox {detection_bbox_abs} (H:{h_det:.0f}, W:{w_det:.0f}) failed MIN_BBOX_SIZE ({self.MIN_BBOX_SIZE}). Skipping.")
                continue
            
            if mask_poly_points is None or mask_poly_points.size == 0:
                logging.debug(f"    BatchExtract Det {det_idx}: No mask polygon points. Skipping.")
                continue

            # Ensure mask_poly_points is int32 for cv2.boundingRect
            mask_poly_points_int = mask_poly_points.astype(np.int32)
            if mask_poly_points_int.shape[0] < 3 : # Need at least 3 points for a polygon
                logging.debug(f"    BatchExtract Det {det_idx}: Not enough points in mask polygon ({mask_poly_points_int.shape[0]}). Skipping.")
                continue

            try:
                x_m, y_m, w_m, h_m = cv2.boundingRect(mask_poly_points_int)
            except cv2.error as e:
                logging.error(f"    BatchExtract Det {det_idx}: cv2.boundingRect error with mask points {mask_poly_points_int}: {e}. Skipping.")
                continue
                
            mask_bbox_for_reid = [x_m, y_m, x_m + w_m, y_m + h_m] 
            logging.debug(f"    BatchExtract Det {det_idx}: Mask BBox for ReID: {mask_bbox_for_reid} (H:{h_m}, W:{w_m})")

            x_m_c = max(0, x_m); y_m_c = max(0, y_m)
            x_m2_c = min(frame_w, x_m + w_m); y_m2_c = min(frame_h, y_m + h_m)
            
            current_patch_w = x_m2_c - x_m_c
            current_patch_h = y_m2_c - y_m_c
            logging.debug(f"    BatchExtract Det {det_idx}: Clipped mask patch H:{current_patch_h}, W:{current_patch_w}")

            if current_patch_w < self.MIN_REID_INPUT_DIM or \
               current_patch_h < self.MIN_REID_INPUT_DIM:
                logging.debug(f"    BatchExtract Det {det_idx}: Mask crop for ReID too small (H:{current_patch_h}, W:{current_patch_w} vs min ReID dim {self.MIN_REID_INPUT_DIM}). Skipping.")
                continue

            person_patch = frame[y_m_c:y_m2_c, x_m_c:x_m2_c]
            if person_patch.size == 0:
                logging.debug(f"    BatchExtract Det {det_idx}: Person patch from mask is empty. Skipping.")
                continue

            local_mask_poly = mask_poly_points_int.copy()
            local_mask_poly[:, 0] -= x_m_c 
            local_mask_poly[:, 1] -= y_m_c 
            
            binary_mask_for_patch = np.zeros((person_patch.shape[0], person_patch.shape[1]), dtype=np.uint8)
            # try:
            #     cv2.fillPoly(binary_mask_for_patch, [local_mask_poly], 255)
            # except cv2.error as e:
            #     logging.error(f"    BatchExtract Det {det_idx}: cv2.fillPoly error with local mask points {local_mask_poly}: {e}. Skipping.")
            #     continue

            masked_person_image = cv2.bitwise_and(person_patch, person_patch, mask=binary_mask_for_patch)
            
            current_person_map_key = len(original_bboxes_and_masks_for_output)
            original_bboxes_and_masks_for_output.append(
                {'bbox_from_mask': mask_bbox_for_reid, 'mask_polygon_orig': mask_poly_points} # Store original float mask points for drawing
            )
            map_list_entry = person_to_crop_indices_map[current_person_map_key]

            added_normal = self._add_crop_to_list(masked_person_image, all_crops_to_extract, map_list_entry)
            added_flipped = self._add_crop_to_list(cv2.flip(masked_person_image, 1), all_crops_to_extract, map_list_entry)
            
            if not (added_normal or added_flipped):
                logging.debug(f"    BatchExtract Det {det_idx}: Neither normal nor flipped masked image was suitable for ReID. Removing from map.")
                if current_person_map_key in person_to_crop_indices_map:
                    del person_to_crop_indices_map[current_person_map_key]
                if original_bboxes_and_masks_for_output and \
                   original_bboxes_and_masks_for_output[-1]['mask_polygon_orig'] is mask_poly_points: # roughly check if it's the one we just added
                    original_bboxes_and_masks_for_output.pop()


        if not all_crops_to_extract:
            logging.debug("BatchExtract: No valid crops to extract features from.")
            return []
        
        logging.debug(f"BatchExtract: Extracting features from {len(all_crops_to_extract)} crops.")
        try:
            features_tensor = self.extractor(all_crops_to_extract)
            features_np = features_tensor.cpu().numpy()
        except Exception as e:
            logging.error(f"Batch feature extraction failed: {str(e)}")
            return []
        
        retained_original_data = []
        final_person_to_crop_map = defaultdict(list)
        new_person_key_idx = 0

        sorted_original_keys = sorted(person_to_crop_indices_map.keys())

        for original_key in sorted_original_keys:
            crop_indices_for_person = person_to_crop_indices_map[original_key]
            if crop_indices_for_person and original_key < len(original_bboxes_and_masks_for_output): # Ensure key is valid
                retained_original_data.append(original_bboxes_and_masks_for_output[original_key])
                final_person_to_crop_map[new_person_key_idx] = crop_indices_for_person
                new_person_key_idx +=1
            else:
                logging.warning(f"BatchExtract: Discrepancy for original_key {original_key} or no crops. Len of original_bboxes_and_masks_for_output: {len(original_bboxes_and_masks_for_output)}")

        original_bboxes_and_masks_for_output = retained_original_data
        person_to_crop_indices_map = final_person_to_crop_map

        for person_key, crop_indices in person_to_crop_indices_map.items():
            if not crop_indices: continue 

            person_feats = [features_np[i] / (np.linalg.norm(features_np[i]) + 1e-9) for i in crop_indices]
            if person_feats:
                agg_feat = np.mean(person_feats, axis=0)
                agg_feat /= (np.linalg.norm(agg_feat) + 1e-9)
                self.feature_buffer_for_cov.append(agg_feat.copy())
                
                if person_key < len(original_bboxes_and_masks_for_output):
                    associated_data = original_bboxes_and_masks_for_output[person_key]
                    bbox_reid = associated_data['bbox_from_mask'] 
                    mask_poly_orig = associated_data['mask_polygon_orig'] 
                    valid_features_info.append((bbox_reid, agg_feat, mask_poly_orig))
                else:
                    logging.warning(f"BatchExtract: person_key {person_key} out of bounds for original_bboxes_and_masks_for_output (len: {len(original_bboxes_and_masks_for_output)}) when creating valid_features_info.")

        logging.debug(f"BatchExtract: Produced {len(valid_features_info)} aggregated features.")
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
        logging.debug(f"--- Processing frame {frame_id} ---")
        self._log_similarity_stats(frame_id)

        if frame_id > 0 and frame_id % self.PRUNE_INTERVAL == 0:
            self.prune_database(frame_id)
            self.save_database()
        if frame_id > 0 and frame_id % self.cov_update_interval == 0:
            self._update_global_covariance_matrix()

        yolo_results_list = self.yolo(frame, classes=0, verbose=False)
        raw_detections_data = []
        logging.debug(f"Frame {frame_id}: YOLO call completed. Results list length: {len(yolo_results_list) if yolo_results_list else 'None'}")

        if yolo_results_list:
            yolo_results = yolo_results_list[0]
            logging.debug(f"Frame {frame_id}: YOLO Results. Has boxes: {yolo_results.boxes is not None}. Has masks: {yolo_results.masks is not None}")

            if yolo_results.boxes is not None:
                logging.debug(f"Frame {frame_id}: Num boxes from YOLO: {len(yolo_results.boxes)}")
            if yolo_results.masks is not None and hasattr(yolo_results.masks, 'xy'):
                 logging.debug(f"Frame {frame_id}: Num masks (from masks.xy): {len(yolo_results.masks.xy) if yolo_results.masks.xy is not None else 'None'}")
            
            if yolo_results.masks is not None and yolo_results.boxes is not None and yolo_results.masks.xy is not None:
                # frame_h, frame_w = frame.shape[:2] # Not needed here, xy is pixel
                logging.debug(f"Frame {frame_id}: Processing {len(yolo_results.boxes)} detected boxes from YOLO.")
                for i in range(len(yolo_results.boxes)):
                    box_obj = yolo_results.boxes[i]
                    # .xyxy are pixel coordinates [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(float, box_obj.xyxy[0])
                    conf = float(box_obj.conf[0])
                    
                    logging.debug(f"  Box {i}: Conf={conf:.2f}, BBox(pixel)=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

                    if conf >= self.DETECTION_CONFIDENCE:
                        logging.debug(f"    Box {i} passed confidence threshold.")
                        if i < len(yolo_results.masks.xy): 
                            # masks.xy provides polygon points in pixel coordinates
                            mask_coords_pixel_from_yolo = yolo_results.masks.xy[i] 
                            
                            if not isinstance(mask_coords_pixel_from_yolo, np.ndarray):
                                mask_coords_pixel_from_yolo = np.array(mask_coords_pixel_from_yolo)

                            # Ensure it's (N, 2) and int32 for cv2 functions
                            mask_coords_pixel_from_yolo = mask_coords_pixel_from_yolo.reshape(-1, 2).astype(np.float32) # Keep as float for now, convert to int later in batch_extract

                            logging.debug(f"      Mask {i} (pixel coords from YOLO): Shape {mask_coords_pixel_from_yolo.shape}, Type {mask_coords_pixel_from_yolo.dtype}")
                                
                            raw_detections_data.append({
                                'bbox': [x1, y1, x2, y2], 
                                'conf': conf, 
                                'mask_polygon': mask_coords_pixel_from_yolo 
                            })
                            logging.debug(f"      Added detection {i} to raw_detections_data. Polygon points: {len(mask_coords_pixel_from_yolo)}")
                        else:
                            logging.warning(f"    Frame {frame_id}: Mask data (index {i}) missing for high-conf detection box {i}. Mask list len: {len(yolo_results.masks.xy)}")
                    else:
                        logging.debug(f"    Box {i} failed confidence threshold ({conf:.2f} < {self.DETECTION_CONFIDENCE}).")
            else:
                logging.debug(f"Frame {frame_id}: YOLO results missing boxes, masks, or masks.xy. Cannot proceed with this frame's detections.")
        
        logging.debug(f"Frame {frame_id}: Number of raw detections after YOLO processing and confidence filter: {len(raw_detections_data)}")
        nms_detections = self._non_max_suppression(raw_detections_data)
        logging.debug(f"Frame {frame_id}: Number of detections after NMS: {len(nms_detections)}")

        current_features_info = self.batch_extract_features(frame, nms_detections) 
        logging.debug(f"Frame {frame_id}: Number of features extracted: {len(current_features_info)}")
        
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
            for det_idx, (bbox_new_mask, _, _) in enumerate(current_features_info):
                if det_idx in assigned_det_indices: continue
                iou_val = self._calculate_iou(bbox_new_mask, track_info['bbox'])
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

            bbox_new_mask, raw_feat, mask_poly_orig = current_features_info[det_idx]
            self.update_track(track_id, bbox_new_mask, raw_feat, frame_id, mask_polygon=mask_poly_orig)
            final_results_this_frame.append((bbox_new_mask, track_id, mask_poly_orig))
            assigned_det_indices.add(det_idx)
            logging.debug(f"Frame {frame_id}: Matched ID {track_id} to det_idx {det_idx} by IOU (score: {iou_score:.2f}). Bbox: {bbox_new_mask}")

        feature_matched_ids_this_frame = set(res[1] for res in final_results_this_frame if res[1] is not None and not isinstance(res[1], str))
        
        for det_idx, (bbox_new_mask, raw_feat, mask_poly_orig) in enumerate(current_features_info):
            if det_idx in assigned_det_indices: continue

            match_id, sim = self.match_features(raw_feat)
            if match_id is not None and match_id not in feature_matched_ids_this_frame:
                self.update_track(match_id, bbox_new_mask, raw_feat, frame_id, mask_polygon=mask_poly_orig)
                final_results_this_frame.append((bbox_new_mask, match_id, mask_poly_orig))
                assigned_det_indices.add(det_idx)
                feature_matched_ids_this_frame.add(match_id) 
                logging.debug(f"Frame {frame_id}: Matched ID {match_id} to det_idx {det_idx} by features (sim: {sim:.2f}). Bbox: {bbox_new_mask}")
        
        for det_idx, (bbox_new_mask, raw_feat, mask_poly_orig) in enumerate(current_features_info):
            if det_idx in assigned_det_indices: continue

            recovered_id, sim_stale = self._match_against_stale_cache(raw_feat, threshold=self.STALE_TRACK_RECOVERY_THRESHOLD)
            if recovered_id is not None and recovered_id not in feature_matched_ids_this_frame:
                stale_mask = self.stale_track_cache.get(recovered_id, {}).get('mask_polygon', mask_poly_orig)
                self.update_track(recovered_id, bbox_new_mask, raw_feat, frame_id, is_recovery=True, mask_polygon=stale_mask)
                final_results_this_frame.append((bbox_new_mask, recovered_id, stale_mask))
                assigned_det_indices.add(det_idx)
                feature_matched_ids_this_frame.add(recovered_id)
                logging.info(f"Frame {frame_id}: Recovered stale ID {recovered_id} for det_idx {det_idx} (sim: {sim_stale:.2f}). Bbox: {bbox_new_mask}")

        next_pending_tracks = {}
        pending_tracks_to_remove_this_step = set()

        for temp_id, p_data in list(self.pending_tracks.items()): 
            p_bbox_current = p_data['display_bbox'] 
            p_feature_history = p_data['feature_history']
            p_first_seen = p_data['first_seen_frame']
            
            found_match_for_pending = False
            best_iou_for_pending, matched_det_idx_for_pending = 0, -1

            for det_idx, (bbox_curr_det_mask, _, _) in enumerate(current_features_info):
                if det_idx in assigned_det_indices: continue 
                iou = self._calculate_iou(p_bbox_current, bbox_curr_det_mask)
                if iou > self.PENDING_IOU_THRESHOLD and iou > best_iou_for_pending:
                    best_iou_for_pending = iou
                    matched_det_idx_for_pending = det_idx
            
            if matched_det_idx_for_pending != -1: 
                found_match_for_pending = True
                assigned_det_indices.add(matched_det_idx_for_pending) 
                
                current_bbox_of_pending, current_raw_feat_of_pending, current_mask_of_pending = current_features_info[matched_det_idx_for_pending]
                
                p_feature_history.append(current_raw_feat_of_pending)
                if 'bbox_history' in p_data: p_data['bbox_history'].append(current_bbox_of_pending)
                p_data['display_bbox'] = current_bbox_of_pending
                p_data['current_mask_polygon'] = current_mask_of_pending 
                p_data['last_seen_frame'] = frame_id
                p_data['missed_frames_count'] = 0 
                
                avg_feat = np.mean(list(p_feature_history), axis=0)
                avg_feat /= (np.linalg.norm(avg_feat) + 1e-9)

                confirmed_id, sim = self.match_features(avg_feat)
                is_stale_recovery = False
                if confirmed_id is None:
                    confirmed_id, sim = self._match_against_stale_cache(avg_feat, threshold=self.STALE_TRACK_RECOVERY_THRESHOLD)
                    if confirmed_id is not None: is_stale_recovery = True
                
                if confirmed_id is not None and confirmed_id not in feature_matched_ids_this_frame:
                    mask_for_confirmed = p_data['current_mask_polygon']
                    self.update_track(confirmed_id, current_bbox_of_pending, avg_feat, frame_id, 
                                      is_recovery=is_stale_recovery, mask_polygon=mask_for_confirmed)
                    final_results_this_frame.append((current_bbox_of_pending, confirmed_id, mask_for_confirmed))
                    feature_matched_ids_this_frame.add(confirmed_id)
                    pending_tracks_to_remove_this_step.add(temp_id)
                    logging.info(f"Frame {frame_id}: Confirmed pending track {temp_id} as {'stale ' if is_stale_recovery else ''}ID {confirmed_id} (sim: {sim:.2f}).")
                else: 
                    final_results_this_frame.append((current_bbox_of_pending, temp_id, p_data['current_mask_polygon']))
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
                mask_for_finalized_pending = p_data.get('current_mask_polygon')

                if final_id is not None and final_id not in feature_matched_ids_this_frame:
                    self.update_track(final_id, current_display_bbox_for_finalized_pending, avg_feat, frame_id, 
                                      is_recovery=is_stale_recovery, mask_polygon=mask_for_finalized_pending)
                    updated_in_results = False
                    for i, res_tuple in enumerate(final_results_this_frame):
                        if res_tuple[1] == temp_id: 
                            final_results_this_frame[i] = (res_tuple[0], final_id, mask_for_finalized_pending)
                            updated_in_results = True; break
                    if not updated_in_results:
                         final_results_this_frame.append((current_display_bbox_for_finalized_pending, final_id, mask_for_finalized_pending))
                    feature_matched_ids_this_frame.add(final_id)
                    logging.info(f"Frame {frame_id}: Pending track {temp_id} expired, confirmed as {'stale ' if is_stale_recovery else ''}ID {final_id} (sim: {sim:.2f}).")
                else: 
                    new_actual_id = self.next_id; self.next_id += 1
                    self.update_track(new_actual_id, current_display_bbox_for_finalized_pending, avg_feat, frame_id, 
                                      mask_polygon=mask_for_finalized_pending)
                    updated_in_results = False
                    for i, res_tuple in enumerate(final_results_this_frame):
                        if res_tuple[1] == temp_id: 
                            final_results_this_frame[i] = (res_tuple[0], new_actual_id, mask_for_finalized_pending)
                            updated_in_results = True; break
                    if not updated_in_results:
                         final_results_this_frame.append((current_display_bbox_for_finalized_pending, new_actual_id, mask_for_finalized_pending))
                    feature_matched_ids_this_frame.add(new_actual_id)
                    logging.info(f"Frame {frame_id}: Pending track {temp_id} expired, assigned new ID {new_actual_id}.")
                pending_tracks_to_remove_this_step.add(temp_id)
            
            elif not found_match_for_pending and temp_id not in pending_tracks_to_remove_this_step:
                if not any(res[1] == temp_id for res in final_results_this_frame):
                    final_results_this_frame.append((p_data['display_bbox'], temp_id, p_data.get('current_mask_polygon')))
                next_pending_tracks[temp_id] = p_data

        self.pending_tracks = {k:v for k,v in next_pending_tracks.items() if k not in pending_tracks_to_remove_this_step}

        for det_idx, (bbox_mask, raw_feat, mask_poly_orig) in enumerate(current_features_info):
            if det_idx in assigned_det_indices: continue

            new_temp_id = f"pending_{self.next_pending_id_counter}"
            self.next_pending_id_counter += 1
            
            self.pending_tracks[new_temp_id] = {
                'display_bbox': bbox_mask, 
                'current_mask_polygon': mask_poly_orig, 
                'feature_history': deque([raw_feat], maxlen=self.WAITING_FRAMES + 5), 
                'bbox_history': deque([bbox_mask], maxlen=self.WAITING_FRAMES + 5),
                'first_seen_frame': frame_id,
                'last_seen_frame': frame_id,
                'missed_frames_count': 0
            }
            final_results_this_frame.append((bbox_mask, new_temp_id, mask_poly_orig))
            logging.info(f"Frame {frame_id}: New detection (det_idx {det_idx}) became pending track {new_temp_id}. Bbox: {bbox_mask}")

        self.cleanup_stale_tracks(frame_id)
        
        logging.debug(f"Frame {frame_id}: Results before final size filter: {len(final_results_this_frame)}")
        filtered_display_results = []
        for bbox_res, id_res, mask_res in final_results_this_frame:
            if self.check_bbox_size(bbox_res): 
                filtered_display_results.append((bbox_res, id_res, mask_res))
            else:
                h_res, w_res = bbox_res[3]-bbox_res[1], bbox_res[2]-bbox_res[0]
                logging.debug(f"Frame {frame_id}: Suppressing display of ID/Pending {id_res} due to small mask bbox: {bbox_res} (H:{h_res:.0f}, W:{w_res:.0f} vs MIN_BBOX_SIZE {self.MIN_BBOX_SIZE})")
        
        logging.debug(f"Frame {frame_id}: Results after final size filter: {len(filtered_display_results)}")
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

    def update_track(self, track_id, bbox_new, feature_to_incorporate, frame_id, mask_polygon=None, is_recovery=False):
        prev_smoothed_feature = None
        
        if is_recovery and track_id in self.stale_track_cache:
            recovered_data = self.stale_track_cache.pop(track_id) 
            prev_smoothed_feature = recovered_data['smoothed_feature']
            current_mask_polygon = recovered_data.get('mask_polygon', mask_polygon) 
            self.detection_history[track_id] = {
                'bbox': recovered_data.get('bbox', bbox_new), 
                'smoothed_feature': prev_smoothed_feature, 
                'velocity': recovered_data.get('velocity', (0.0,0.0)),
                'frames_missing': 0, 
                'last_seen_frame': recovered_data.get('last_seen_frame_active', frame_id),
                'mask_polygon': current_mask_polygon 
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
            'last_seen_frame': frame_id,
            'mask_polygon': mask_polygon 
        })
        self.update_feature_store(track_id, new_smoothed_feature, frame_id) 
        self.last_active[track_id] = frame_id 

    def cleanup_stale_tracks(self, frame_id): 
        stale_ids_to_move = []
        for track_id, track_data in list(self.detection_history.items()):
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
                        'mask_polygon': track_to_stale.get('mask_polygon'), 
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
        min_bbox_size=32, # Lowered for testing, adjust as needed for your video
        detection_confidence=0.3 # Lowered for testing
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
            
            if frame is None:
                logging.warning(f"Frame {video_frame_num} from {video_path.name} is None. Skipping.")
                video_frame_num +=1
                global_frame_counter +=1
                continue

            processed_results = reid_system.process_frame(frame.copy(), global_frame_counter)
            
            global_frame_counter += 1
            video_frame_num +=1
            
            display_frame_with_masks = frame.copy()

            for bbox_mask, obj_id, mask_polygon in processed_results:
                x1, y1, x2, y2 = map(int, bbox_mask) 
                
                color = (0, 255, 0) 
                label_prefix = "ID: "
                if isinstance(obj_id, str) and obj_id.startswith("pending_"):
                    color = (0, 255, 255) 
                    label_prefix = "Waiting: " 
                    obj_id_display = '' 
                else:
                    obj_id_display = str(obj_id)

                cv2.rectangle(display_frame_with_masks, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame_with_masks, f"{label_prefix}{obj_id_display}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                if mask_polygon is not None and mask_polygon.size > 0:
                    overlay = display_frame_with_masks.copy()
                    cv2.fillPoly(overlay, [mask_polygon.astype(np.int32)], color) # Ensure int32 for fillPoly
                    alpha = 0.3 
                    cv2.addWeighted(overlay, alpha, display_frame_with_masks, 1 - alpha, 0, display_frame_with_masks)
            
            info_active = f"Active: {len(reid_system.detection_history)}"
            info_pending = f"Pending: {len(reid_system.pending_tracks)}"
            info_stale = f"Stale: {len(reid_system.stale_track_cache)}"
            info_db = f"DB IDs: {len(reid_system.feature_db)}"
            info_next_id = f"NextID: {reid_system.next_id}"
            
            cv2.putText(display_frame_with_masks, f"{video_path.name} (F:{video_frame_num})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(display_frame_with_masks, info_active, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)
            cv2.putText(display_frame_with_masks, info_pending, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,255), 2) 
            cv2.putText(display_frame_with_masks, info_stale, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
            cv2.putText(display_frame_with_masks, info_db, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,200), 2)
            cv2.putText(display_frame_with_masks, info_next_id, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,200), 2)

            if target_display_width > 0 and target_display_height > 0:
                final_display = cv2.resize(display_frame_with_masks, (target_display_width, target_display_height))
            else:
                final_display = display_frame_with_masks

            cv2.imshow('Person Re-ID (Segmentation-based)', final_display)
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
        './videos/15-06-25--1.mp4', # Replace with your video file
      
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