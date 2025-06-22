import json
import logging
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from collections import defaultdict
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PersonReID:
    def __init__(self,
                 yolo_model='yolo11m-seg.onnx',
                 reid_model='./models/osnet_ain_x1_0_market1501_256x128_amsgrad_ep100_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth',
                 device='cuda',
                 waiting_frames=4, # You set this to 1
                 min_bbox_size=90,
                 track_iou_threshold=0.99,
                 feature_match_threshold=0.76,
                 max_frames_missing=120,
                 detection_confidence=0.6, # You set this to 0.6
                 nms_pre_reid_iou_threshold=0.99,
                 cluster_merge_threshold=0.76,
                 max_clusters_per_id=8,
                 prune_db_interval=10000,
                 max_inactive_frames_for_db_prune=5000,
                 debug_show_reid_crops=False
                 ):

        self.device = device
        self.yolo = YOLO(yolo_model)
        self.extractor = FeatureExtractor(model_name=reid_model, device=device)

        self.next_id = 1
        self.detection_history = {}
        self.waiting_detections = defaultdict(list)
        self.feature_db = {}
        self.last_active = {}
        self.db_path = Path('reid_database.json')

        self.WAITING_FRAMES = waiting_frames
        self.MIN_BBOX_SIZE = min_bbox_size
        self.TRACK_IOU_THRESHOLD = track_iou_threshold
        self.FEATURE_MATCH_THRESHOLD = feature_match_threshold
        self.MAX_FRAMES_MISSING = max_frames_missing
        self.DETECTION_CONFIDENCE = detection_confidence
        self.NMS_PRE_REID_IOU_THRESHOLD = nms_pre_reid_iou_threshold

        self.CLUSTER_MERGE_THRESHOLD = cluster_merge_threshold
        self.MAX_CLUSTERS_PER_ID = max_clusters_per_id
        self.PRUNE_DB_INTERVAL = prune_db_interval
        self.MAX_INACTIVE_FRAMES_FOR_DB_PRUNE = max_inactive_frames_for_db_prune
        self.DEBUG_SHOW_REID_CROPS = debug_show_reid_crops

        self.previous_detection_count = 0
        self.detection_change_frame = 0 # Frame ID when detection count last changed
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
                                if centroid.ndim > 1:
                                    centroid = centroid.flatten()

                                expected_dim = 512 # For OSNet
                                if hasattr(self.extractor, 'feature_dim') and self.extractor.feature_dim:
                                     expected_dim = self.extractor.feature_dim

                                if centroid.ndim == 1 and centroid.shape[0] == expected_dim:
                                    clusters.append({
                                        'centroid': centroid,
                                        'count': int(c_data['count'])
                                    })
                                else:
                                    logging.warning(f"Skipping cluster for ID {obj_id} due to unexpected feature dimension: {centroid.shape}, expected {expected_dim}")
                            else:
                                logging.warning(f"Skipping malformed cluster data for ID {obj_id_str}: {c_data}")

                        if clusters:
                           self.feature_db[obj_id] = clusters
                           valid_ids_loaded.append(obj_id)

                    except (ValueError, TypeError) as e:
                        logging.warning(f"Skipping invalid entry {obj_id_str} in database during load: {str(e)}")

                if valid_ids_loaded:
                    self.next_id = max(valid_ids_loaded) + 1
                else:
                    self.next_id = 1
                logging.info(f"Loaded database with {len(self.feature_db)} identities. Next ID: {self.next_id}")
            else:
                logging.info("No existing database found. Starting fresh.")

        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from database file: {str(e)}. Initializing empty database.")
        except Exception as e:
            logging.error(f"Critical error loading database: {str(e)}. Initializing empty database.")
        finally:
            if not hasattr(self, 'feature_db') or not isinstance(self.feature_db, dict):
                 self.feature_db = {}
            if not hasattr(self, 'next_id') or not isinstance(self.next_id, int):
                 self.next_id = 1

        self.last_active = {obj_id: 0 for obj_id in self.feature_db.keys()}


    def save_database(self):
        try:
            temp_path = self.db_path.with_suffix('.tmp')
            serialized_db = {}
            for obj_id, clusters in self.feature_db.items():
                valid_clusters_to_save = []
                for c in clusters:
                    if isinstance(c.get('centroid'), np.ndarray) and isinstance(c.get('count'), int):
                        centroid_list = c['centroid'].astype(float).tolist()
                        valid_clusters_to_save.append({'centroid': centroid_list, 'count': c['count']})
                    else:
                        logging.warning(f"Skipping saving malformed cluster for ID {obj_id} in save_database: {c}")
                if valid_clusters_to_save:
                    serialized_db[str(obj_id)] = valid_clusters_to_save

            with open(temp_path, 'w') as f:
                json.dump(serialized_db, f, indent=2)
            temp_path.replace(self.db_path)
            logging.info(f"Database saved with {len(serialized_db)} identities to {self.db_path}")

        except Exception as e:
            logging.error(f"Database save failed: {str(e)}")

    def reset_video_states(self):
        self.detection_history.clear()
        self.waiting_detections.clear()
        self.previous_detection_count = 0
        self.detection_change_frame = 0
        self.is_in_waiting_period = False
        logging.info("Per-video tracking states (detection history, waiting period) have been reset.")

    def _calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / (union_area + 1e-9)
        return iou

    def _non_max_suppression(self, boxes_data, iou_threshold):
        if not boxes_data:
            return []

        boxes_data = sorted(boxes_data, key=lambda x: x['conf'], reverse=True)
        selected_boxes_data = []

        while boxes_data:
            chosen_box_data = boxes_data.pop(0)
            selected_boxes_data.append(chosen_box_data)

            remaining_boxes_data = []
            for box_data in boxes_data:
                iou = self._calculate_iou(chosen_box_data['bbox'], box_data['bbox'])
                if iou < iou_threshold:
                    remaining_boxes_data.append(box_data)
            boxes_data = remaining_boxes_data

        return selected_boxes_data

    def check_bbox_size(self, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0:
            return False
        return h >= self.MIN_BBOX_SIZE and w >= self.MIN_BBOX_SIZE

    def batch_extract_features(self, frame, nms_detections_data, frame_id_for_log="N/A"):
        valid_features_info = []
        all_crops_to_extract = []
        detection_to_feature_indices_map = defaultdict(list)
        original_bboxes_for_output = []

        for det_idx, det_data in enumerate(nms_detections_data):
            bbox_abs = det_data['bbox']
            mask_polygon_abs = det_data['mask']

            if not self.check_bbox_size(bbox_abs):
                continue

            x1_abs, y1_abs, x2_abs, y2_abs = map(int, bbox_abs)
            person_bbox_crop = frame[y1_abs:y2_abs, x1_abs:x2_abs]

            if person_bbox_crop.size == 0 or person_bbox_crop.shape[0] < 10 or person_bbox_crop.shape[1] < 10:
                continue

            masked_person_image_for_reid = None
            if mask_polygon_abs is not None and isinstance(mask_polygon_abs, np.ndarray) and mask_polygon_abs.ndim == 2 and mask_polygon_abs.shape[0] > 0:
                h_crop, w_crop = person_bbox_crop.shape[:2]

                mask_polygon_relative_to_crop = mask_polygon_abs.copy().astype(np.float32)
                mask_polygon_relative_to_crop[:, 0] -= x1_abs
                mask_polygon_relative_to_crop[:, 1] -= y1_abs

                mask_polygon_relative_to_crop[:, 0] = np.clip(mask_polygon_relative_to_crop[:, 0], 0, w_crop - 1)
                mask_polygon_relative_to_crop[:, 1] = np.clip(mask_polygon_relative_to_crop[:, 1], 0, h_crop - 1)

                binary_segment_mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
                cv2.fillPoly(binary_segment_mask, [mask_polygon_relative_to_crop.astype(np.int32)], 255)

                masked_person_image_for_reid = person_bbox_crop.copy()
                masked_person_image_for_reid[binary_segment_mask == 0] = [0, 0, 0]
            else:
                logging.warning(f"Frame {frame_id_for_log}: Valid mask not available for detection at {list(map(int, bbox_abs))}. Using bbox crop.")
                masked_person_image_for_reid = person_bbox_crop.copy() # Ensure it's a copy

            if masked_person_image_for_reid.size == 0 or masked_person_image_for_reid.shape[0] < 5 or masked_person_image_for_reid.shape[1] < 5:
                continue

            original_bboxes_for_output.append(bbox_abs)
            current_detection_output_idx = len(original_bboxes_for_output) - 1

            if self.DEBUG_SHOW_REID_CROPS:
                try:
                    if masked_person_image_for_reid.shape[0] > 10 and masked_person_image_for_reid.shape[1] > 10:
                        display_crop = cv2.resize(masked_person_image_for_reid, (128, 256), interpolation=cv2.INTER_NEAREST)
                        cv2.imshow(f"ReID Crop Det {det_idx} (F{frame_id_for_log})", display_crop)
                except Exception as e_vis:
                    logging.warning(f"Could not display ReID crop for det {det_idx}: {e_vis}")

            all_crops_to_extract.append(masked_person_image_for_reid)
            detection_to_feature_indices_map[current_detection_output_idx].append(len(all_crops_to_extract) - 1)

            flipped_crop = cv2.flip(masked_person_image_for_reid, 1)
            if self.DEBUG_SHOW_REID_CROPS:
                try:
                     if flipped_crop.shape[0] > 10 and flipped_crop.shape[1] > 10:
                        display_flipped_crop = cv2.resize(flipped_crop, (128, 256), interpolation=cv2.INTER_NEAREST)
                        cv2.imshow(f"ReID FlipDet {det_idx} (F{frame_id_for_log})", display_flipped_crop)
                except Exception as e_vis:
                    logging.warning(f"Could not display flipped ReID crop for det {det_idx}: {e_vis}")
            all_crops_to_extract.append(flipped_crop)
            detection_to_feature_indices_map[current_detection_output_idx].append(len(all_crops_to_extract) - 1)


        if not all_crops_to_extract:
            return []

        try:
            extracted_features_tensor = self.extractor(all_crops_to_extract)
            extracted_features_tensor = torch.nn.functional.normalize(extracted_features_tensor, p=2, dim=1)
            extracted_features_np = extracted_features_tensor.cpu().numpy()
        except Exception as e:
            logging.error(f"Feature extraction batch processing failed: {str(e)}")
            return []

        for det_output_idx, feature_indices_in_batch in detection_to_feature_indices_map.items():
            if not feature_indices_in_batch:
                continue
            person_features_list = [extracted_features_np[crop_idx] for crop_idx in feature_indices_in_batch]
            if person_features_list:
                aggregated_feature = np.mean(person_features_list, axis=0)
                aggregated_feature /= (np.linalg.norm(aggregated_feature) + 1e-9)
                if det_output_idx < len(original_bboxes_for_output):
                    original_bbox = original_bboxes_for_output[det_output_idx]
                    valid_features_info.append((original_bbox, aggregated_feature))
                else:
                    logging.warning(f"Mismatch in det_output_idx {det_output_idx} for feature aggregation. Skipping.")
        return valid_features_info

    def match_features(self, query_feature, threshold=None):
        match_threshold = threshold if threshold is not None else self.FEATURE_MATCH_THRESHOLD
        if not self.feature_db or query_feature is None:
            return None, 0.0
        best_match_id = None
        highest_similarity = -1.0
        for obj_id, clusters in self.feature_db.items():
            if not clusters: continue
            max_sim_for_this_id = -1.0
            for cluster in clusters:
                similarity = np.dot(query_feature, cluster['centroid'])
                if similarity > max_sim_for_this_id:
                    max_sim_for_this_id = similarity
            if max_sim_for_this_id > highest_similarity:
                highest_similarity = max_sim_for_this_id
                best_match_id = obj_id
        if best_match_id is not None and highest_similarity >= match_threshold:
            return best_match_id, highest_similarity
        else:
            return None, highest_similarity

    def update_feature_store(self, obj_id, new_feature, frame_id):
        if obj_id not in self.feature_db:
            self.feature_db[obj_id] = [{'centroid': new_feature, 'count': 1}]
        else:
            clusters = self.feature_db[obj_id]
            best_sim_to_cluster = -1.0
            best_cluster_idx = -1
            for i, cluster in enumerate(clusters):
                sim = np.dot(new_feature, cluster['centroid'])
                if sim > best_sim_to_cluster:
                    best_sim_to_cluster = sim
                    best_cluster_idx = i
            if best_sim_to_cluster >= self.CLUSTER_MERGE_THRESHOLD and best_cluster_idx != -1:
                matched_cluster = clusters[best_cluster_idx]
                old_centroid = matched_cluster['centroid']
                old_count = matched_cluster['count']
                new_centroid_unnormalized = (old_centroid * old_count + new_feature) / (old_count + 1)
                matched_cluster['centroid'] = new_centroid_unnormalized / (np.linalg.norm(new_centroid_unnormalized) + 1e-9)
                matched_cluster['count'] += 1
            else:
                clusters.append({'centroid': new_feature, 'count': 1})
                if len(clusters) > self.MAX_CLUSTERS_PER_ID:
                    clusters.sort(key=lambda c: c['count'])
                    del clusters[0]
        self.last_active[obj_id] = frame_id

    def prune_database(self, current_frame_id):
        if not self.feature_db: return
        ids_to_prune = [
            obj_id for obj_id, last_seen_frame in self.last_active.items()
            if current_frame_id - last_seen_frame > self.MAX_INACTIVE_FRAMES_FOR_DB_PRUNE
        ]
        pruned_count = 0
        for obj_id in ids_to_prune:
            if obj_id in self.feature_db: del self.feature_db[obj_id]; pruned_count +=1
            if obj_id in self.last_active: del self.last_active[obj_id]
        if pruned_count > 0:
            logging.info(f"Pruned {pruned_count} inactive IDs from feature_db.")

    def manage_tracking_states(self, current_detections_count, frame_id):
        count_changed = current_detections_count != self.previous_detection_count
        if count_changed:
            if not self.is_in_waiting_period : # Only enter if not already in waiting period
                self.is_in_waiting_period = True
                self.detection_change_frame = frame_id
                self.waiting_detections.clear()
                logging.debug(f"Frame {frame_id}: Detection count changed ({self.previous_detection_count} -> {current_detections_count}). Entering waiting for {self.WAITING_FRAMES} frames.")
        self.previous_detection_count = current_detections_count
        # The decision to exit waiting period is now handled in process_frame

    def process_frame(self, frame, frame_id):
        if frame_id > 0 and frame_id % self.PRUNE_DB_INTERVAL == 0:
            self.prune_database(frame_id)
            self.save_database()

        yolo_results_list = self.yolo(frame, classes=0, verbose=False, conf=self.DETECTION_CONFIDENCE)
        
        current_detections_count_from_yolo = 0
        raw_detections_data = []

        if yolo_results_list:
            yolo_results = yolo_results_list[0]
            if hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
                num_boxes = len(yolo_results.boxes)
                current_detections_count_from_yolo = num_boxes # Count before any filtering by size etc.
                
                masks_xy_list = None
                if hasattr(yolo_results, 'masks') and yolo_results.masks is not None and \
                   hasattr(yolo_results.masks, 'xy') and yolo_results.masks.xy is not None:
                    masks_xy_list = yolo_results.masks.xy

                for i in range(num_boxes):
                    box_data_tensor = yolo_results.boxes[i].data[0]
                    box_data = box_data_tensor.cpu().tolist() if isinstance(box_data_tensor, torch.Tensor) else list(box_data_tensor)
                    x1, y1, x2, y2, conf, cls_id = box_data
                    current_mask_polygon = None
                    if masks_xy_list is not None and i < len(masks_xy_list):
                        polygon_coords_np = masks_xy_list[i]
                        if isinstance(polygon_coords_np, np.ndarray) and polygon_coords_np.ndim == 2 and polygon_coords_np.shape[1] == 2:
                            current_mask_polygon = polygon_coords_np
                        else:
                            logging.warning(f"Frame {frame_id}: Mask xy[{i}] has unexpected format.")
                    raw_detections_data.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'mask': current_mask_polygon})
            else:
                logging.warning(f"Frame {frame_id}: No boxes found in YOLO results object.")
        else:
            logging.warning(f"Frame {frame_id}: YOLO returned no results list.")

        # Manage state based on raw YOLO detections before NMS and feature extraction
        self.manage_tracking_states(current_detections_count_from_yolo, frame_id)

        nms_filtered_detections = self._non_max_suppression(raw_detections_data, self.NMS_PRE_REID_IOU_THRESHOLD)
        # Note: current_frame_features_info might be less than current_detections_count_from_yolo due to check_bbox_size in batch_extract_features
        current_frame_features_info = self.batch_extract_features(frame, nms_filtered_detections, frame_id_for_log=frame_id)

        for track_id in list(self.detection_history.keys()):
            if track_id in self.detection_history:
                self.detection_history[track_id]['frames_missing'] += 1
        self.cleanup_stale_tracks(frame_id)

        output_results = []
        waiting_period_should_end = self.is_in_waiting_period and \
                                    (frame_id - self.detection_change_frame >= self.WAITING_FRAMES)

        if waiting_period_should_end:
            logging.debug(f"Frame {frame_id}: Waiting period duration met ({frame_id} - {self.detection_change_frame} >= {self.WAITING_FRAMES}). Processing buffered detections.")
            if self.waiting_detections:
                confirmed_from_waiting = self.process_buffered_waiting_detections(frame_id)
                output_results.extend(confirmed_from_waiting)
            self.is_in_waiting_period = False # Exit waiting period

        if self.is_in_waiting_period: # Still in waiting (duration not met or just entered)
            # Buffer current_frame_features_info, not raw nms_filtered_detections
            output_results.extend(self.buffer_detections_during_waiting(current_frame_features_info, frame_id))
        else: # Not in waiting period (either never entered, or just exited)
            # Process current frame's detections normally
            normal_processed = self.process_normal_detections(current_frame_features_info, frame_id)
            
            # Merge results: normal_processed should ideally update/override any existing results
            temp_final_map = {tuple(map(int, r[0])): r[1] for r in output_results} # Start with results from waiting (if any)
            for bbox, obj_id in normal_processed:
                temp_final_map[tuple(map(int, bbox))] = obj_id 
            output_results = [(list(bbox_tuple), obj_id) for bbox_tuple, obj_id in temp_final_map.items()]
        
        return output_results

    def handle_initial_detections(self, current_features_info, frame_id):
        # This method might become less critical if waiting period and normal processing cover initial states
        results = []
        for bbox, features in current_features_info:
            new_id = self.next_id
            self.next_id += 1
            self.update_track(new_id, bbox, features, frame_id)
            results.append((bbox, new_id))
            logging.info(f"Frame {frame_id}: Initial new ID {new_id} assigned to bbox {list(map(int,bbox))}.")
        return results

    def buffer_detections_during_waiting(self, current_features_info, frame_id):
        # This now returns list of (bbox, None) for display, and internally buffers
        temp_results_for_display = []
        for bbox, features in current_features_info:
            detection_key = tuple(map(int, bbox)) # Using bbox as key for simplicity
            self.waiting_detections[detection_key].append({
                'frame_id': frame_id,
                'features': features,
                'bbox': bbox
            })
            temp_results_for_display.append((bbox, None)) # ID is None during waiting
        if temp_results_for_display:
            logging.debug(f"Frame {frame_id}: Buffered {len(temp_results_for_display)} detections during waiting.")
        return temp_results_for_display


    def process_buffered_waiting_detections(self, frame_id):
        logging.debug(f"Frame {frame_id}: Processing {len(self.waiting_detections)} unique bbox keys from waiting period.")
        confirmed_results = []
        processed_keys = [] # Keep track of keys processed to remove them

        for det_key_bbox_tuple, history_list in self.waiting_detections.items():
            if not history_list:
                processed_keys.append(det_key_bbox_tuple)
                continue

            # Require some persistence within the waiting window
            # For WAITING_FRAMES=1, this means it must have been seen in that 1 frame.
            # For WAITING_FRAMES=6, it needs to be seen in at least ~2 frames (30%).
            min_sighting_frames = max(1, int(self.WAITING_FRAMES * 0.30))
            if len(history_list) < min_sighting_frames:
                logging.debug(f"Skipping buffered detection {det_key_bbox_tuple}, insufficient persistence ({len(history_list)} < {min_sighting_frames} frames).")
                processed_keys.append(det_key_bbox_tuple)
                continue

            try:
                latest_sighting = sorted(history_list, key=lambda x: x['frame_id'])[-1]
                avg_features_list = [item['features'] for item in history_list if item['features'] is not None]
                if not avg_features_list:
                    processed_keys.append(det_key_bbox_tuple)
                    continue

                avg_features = np.mean(avg_features_list, axis=0)
                avg_features /= (np.linalg.norm(avg_features) + 1e-9)

                # Use a slightly relaxed threshold for matching after waiting
                match_id, similarity = self.match_features(avg_features, threshold=self.FEATURE_MATCH_THRESHOLD * 0.90)

                final_id_for_buffered = None
                if match_id is not None:
                    final_id_for_buffered = match_id
                    logging.debug(f"Frame {frame_id}: Buffered detection {list(map(int,latest_sighting['bbox']))} matched existing ID {match_id} (sim: {similarity:.2f}) after waiting.")
                else:
                    final_id_for_buffered = self.next_id
                    self.next_id += 1
                    logging.info(f"Frame {frame_id}: Buffered detection {list(map(int,latest_sighting['bbox']))} confirmed as new ID {final_id_for_buffered} (best sim: {similarity:.2f}) after waiting.")
                
                self.update_track(final_id_for_buffered, latest_sighting['bbox'], avg_features, latest_sighting['frame_id'])
                confirmed_results.append((latest_sighting['bbox'], final_id_for_buffered))
            except Exception as e:
                logging.error(f"Error processing a buffered detection {det_key_bbox_tuple}: {str(e)}")
            finally:
                processed_keys.append(det_key_bbox_tuple)
        
        for key_to_del in processed_keys: # Clear processed items from waiting_detections
            if key_to_del in self.waiting_detections:
                del self.waiting_detections[key_to_del]
        
        if not self.waiting_detections: # Double check if it's truly empty
             logging.debug(f"Frame {frame_id}: waiting_detections buffer is now empty after processing.")
        else:
             logging.warning(f"Frame {frame_id}: waiting_detections buffer still has {len(self.waiting_detections)} items after processing.")


        return confirmed_results

    def process_normal_detections(self, current_features_info, frame_id):
        if not current_features_info and not self.detection_history: # No current detections and no history, nothing to do
            return []
            
        assigned_current_detection_indices = set()
        final_results_this_frame = []

        active_tracks_for_iou = []
        for track_id, track_data in self.detection_history.items():
            if track_data['frames_missing'] <= 1 :
                active_tracks_for_iou.append({
                    'id': track_id,
                    'bbox': track_data['bbox']
                })

        potential_iou_matches = []
        for i, track_info in enumerate(active_tracks_for_iou):
            track_id = track_info['id']
            track_bbox = track_info['bbox']
            best_iou_for_this_track = 0
            best_det_idx_for_this_track = -1
            for det_idx, (det_bbox, _) in enumerate(current_features_info):
                if det_idx in assigned_current_detection_indices: continue
                iou = self._calculate_iou(det_bbox, track_bbox)
                if iou >= self.TRACK_IOU_THRESHOLD and iou > best_iou_for_this_track:
                    best_iou_for_this_track = iou
                    best_det_idx_for_this_track = det_idx
            if best_det_idx_for_this_track != -1:
                potential_iou_matches.append((track_id, best_det_idx_for_this_track, best_iou_for_this_track))

        potential_iou_matches.sort(key=lambda x: x[2], reverse=True)
        ids_assigned_this_frame_iou = set()
        for track_id, det_idx, iou_score in potential_iou_matches:
            if det_idx in assigned_current_detection_indices: continue
            if track_id in ids_assigned_this_frame_iou: continue
            bbox_new, features_new = current_features_info[det_idx]
            self.update_track(track_id, bbox_new, features_new, frame_id)
            final_results_this_frame.append((bbox_new, track_id))
            assigned_current_detection_indices.add(det_idx)
            ids_assigned_this_frame_iou.add(track_id)
            logging.debug(f"Frame {frame_id}: Matched ID {track_id} to detection {det_idx} by IOU (score: {iou_score:.2f}).")

        unmatched_after_iou_indices = [i for i, _ in enumerate(current_features_info) if i not in assigned_current_detection_indices]
        potential_feature_matches = []
        for det_idx in unmatched_after_iou_indices:
            bbox_new, features_new = current_features_info[det_idx]
            if features_new is None: continue
            match_id, similarity = self.match_features(features_new)
            if match_id is not None and match_id not in ids_assigned_this_frame_iou :
                potential_feature_matches.append({'det_idx': det_idx, 'match_id': match_id, 'similarity': similarity, 'bbox': bbox_new, 'features': features_new})
        
        potential_feature_matches.sort(key=lambda x: x['similarity'], reverse=True)
        ids_assigned_this_frame_feature = set()
        for match_info in potential_feature_matches:
            det_idx = match_info['det_idx']
            if det_idx in assigned_current_detection_indices: continue
            match_id = match_info['match_id']
            if match_id in ids_assigned_this_frame_feature or match_id in ids_assigned_this_frame_iou: continue
            self.update_track(match_id, match_info['bbox'], match_info['features'], frame_id)
            final_results_this_frame.append((match_info['bbox'], match_id))
            assigned_current_detection_indices.add(det_idx)
            ids_assigned_this_frame_feature.add(match_id)
            logging.debug(f"Frame {frame_id}: Matched ID {match_id} to detection {det_idx} by features (sim: {match_info['similarity']:.2f}).")

        unmatched_after_all_matching_indices = [i for i, _ in enumerate(current_features_info) if i not in assigned_current_detection_indices]
        for det_idx in unmatched_after_all_matching_indices:
            bbox_new, features_new = current_features_info[det_idx]
            if features_new is None: continue
            new_id = self.next_id
            self.next_id += 1
            self.update_track(new_id, bbox_new, features_new, frame_id)
            final_results_this_frame.append((bbox_new, new_id))
            logging.info(f"Frame {frame_id}: Created new ID {new_id} for unmatched detection {det_idx} at {list(map(int,bbox_new))}.")
        return final_results_this_frame


    def predict_bbox(self, bbox, velocity):
        dx, dy = velocity
        return [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]

    def update_track(self, track_id, bbox_new, features_new, frame_id):
        if features_new is None:
            logging.warning(f"Frame {frame_id}: Attempted to update track {track_id} with None features.")
            if track_id in self.detection_history:
                 self.detection_history[track_id]['frames_missing'] = 0
                 self.detection_history[track_id]['last_seen_frame'] = frame_id
                 self.detection_history[track_id]['bbox'] = bbox_new
            return

        prev_bbox_data = self.detection_history.get(track_id)
        prev_cx = (prev_bbox_data['bbox'][0] + prev_bbox_data['bbox'][2]) / 2 if prev_bbox_data else (bbox_new[0] + bbox_new[2]) / 2
        prev_cy = (prev_bbox_data['bbox'][1] + prev_bbox_data['bbox'][3]) / 2 if prev_bbox_data else (bbox_new[1] + bbox_new[3]) / 2
        new_cx = (bbox_new[0] + bbox_new[2]) / 2
        new_cy = (bbox_new[1] + bbox_new[3]) / 2
        current_dx = new_cx - prev_cx
        current_dy = new_cy - prev_cy
        alpha = 0.5
        prev_vx, prev_vy = prev_bbox_data.get('velocity', (0.0, 0.0)) if prev_bbox_data else (0.0, 0.0)
        vx = alpha * current_dx + (1 - alpha) * prev_vx
        vy = alpha * current_dy + (1 - alpha) * prev_vy
        self.detection_history[track_id] = {
            'bbox': bbox_new, 'features': features_new, 'velocity': (vx, vy),
            'frames_missing': 0, 'last_seen_frame': frame_id
        }
        self.update_feature_store(track_id, features_new, frame_id)

    def cleanup_stale_tracks(self, frame_id):
        stale_ids = [
            track_id for track_id, track_data in self.detection_history.items()
            if track_data['frames_missing'] > self.MAX_FRAMES_MISSING
        ]
        for track_id in stale_ids:
            if track_id in self.detection_history:
                del self.detection_history[track_id]
                logging.info(f"Frame {frame_id}: Removed stale track ID {track_id}.")

def main(video_paths):
    reid_system = PersonReID(
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    global_frame_counter = 0
    for video_idx, video_path_str in enumerate(video_paths):
        video_path = Path(video_path_str)
        logging.info(f"--- Starting processing of video {video_idx + 1}/{len(video_paths)}: {video_path.name} ---")
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logging.error(f"Failed to open video: {video_path}"); continue
        except Exception as e:
            logging.error(f"Error opening video {video_path}: {e}"); continue
        reid_system.reset_video_states()
        video_frame_num = 0
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    logging.info(f"Finished processing video: {video_path.name}"); break
                frame_for_viz = frame.copy()
                results = reid_system.process_frame(frame, global_frame_counter)
                global_frame_counter += 1
                video_frame_num +=1

                # Optional: Draw raw YOLO masks on main display (can be slow)
                # if True: # Always draw masks
                #     yolo_results_list_viz = reid_system.yolo(frame_for_viz, classes=0, verbose=False, conf=reid_system.DETECTION_CONFIDENCE)
                #     if yolo_results_list_viz:
                #         yolo_results_viz = yolo_results_list_viz[0]
                #         if hasattr(yolo_results_viz, 'masks') and yolo_results_viz.masks is not None and \
                #            hasattr(yolo_results_viz.masks, 'xy') and yolo_results_viz.masks.xy is not None:
                #             for polygon_np in yolo_results_viz.masks.xy:
                #                 cv2.polylines(frame_for_viz, [polygon_np.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=1)

                for bbox, obj_id in results:
                    x1, y1, x2, y2 = map(int, bbox)
                    color_seed = obj_id * 30 if obj_id is not None else 0
                    color = (color_seed % 200 + 55, (color_seed * 2) % 200 + 55, (color_seed * 3) % 200 + 55) if obj_id is not None else (0, 255, 255)
                    label = f"ID: {obj_id}" if obj_id is not None else "Waiting..."
                    cv2.rectangle(frame_for_viz, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_for_viz, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                info_text = f"Video: {video_path.name} (F: {video_frame_num}) GF: {global_frame_counter}"
                cv2.putText(frame_for_viz, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                
                H_viz, W_viz = frame_for_viz.shape[:2]
                scale_viz = 1280 / W_viz if W_viz > 1280 else 1.0 # Max width 1280
                if H_viz * scale_viz > 720: scale_viz = 720 / H_viz # Max height 720
                display_frame_viz = cv2.resize(frame_for_viz, (int(W_viz * scale_viz), int(H_viz * scale_viz)))
                cv2.imshow('Person Re-ID', display_frame_viz)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("Q pressed. Exiting."); cap.release(); cv2.destroyAllWindows(); reid_system.save_database(); return
                elif key == ord('p'):
                    logging.info("P pressed. Paused. Press P to resume, Q to quit, any other key to step.")
                    while True:
                        resume_key = cv2.waitKey(0) & 0xFF
                        if resume_key == ord('p'): logging.info("Resuming..."); break
                        elif resume_key == ord('q'): logging.info("Q pressed during pause. Exiting."); cap.release(); cv2.destroyAllWindows(); reid_system.save_database(); return
                        else: logging.info("Stepping one frame..."); break 
            except Exception as e:
                logging.error(f"Error in frame {video_frame_num} of {video_path.name}: {e}", exc_info=True); break
        cap.release()
    cv2.destroyAllWindows()
    reid_system.save_database()
    logging.info("--- All videos processed. Database saved. ---")

if __name__ == "__main__":
    video_files_input = [
         './videos/20-05-1.MOV',
         './videos/20-05-4.MOV'
    ]
    valid_video_paths = []
    if not video_files_input: logging.warning("Video paths list is empty.")
    else:
        for p_str in video_files_input:
            p_path = Path(p_str)
            if p_path.exists() and p_path.is_file(): valid_video_paths.append(str(p_path))
            else: logging.warning(f"Video file not found or is not a file: {p_str}. Skipping.")
    if not valid_video_paths: logging.error("No valid video paths found.")
    else: main(valid_video_paths)