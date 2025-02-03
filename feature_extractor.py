# feature_manager.py
import json
from pathlib import Path
import numpy as np
from config import ReIDConfig

class FeatureManager:
    def __init__(self, config: ReIDConfig, db_path: Path = Path('reid_database.json')):
        self.config = config
        self.db_path = db_path
        self.feature_db = {}
        self.next_id = 1
        self.load_database()

    def load_database(self):
        self.feature_db = {}
        self.next_id = 1
        
        if self.db_path.exists():
            self.db_path.unlink()
            
        try:
            with open(self.db_path, 'w') as f:
                json.dump({}, f)
        except Exception as e:
            print(f"Warning: Could not create database file ({str(e)})")
            self.feature_db = {}

    def save_database(self):
        try:
            serializable_db = {str(k): [v.tolist() for v in vectors] 
                             for k, vectors in self.feature_db.items()}
            with open(self.db_path, 'w') as f:
                json.dump(serializable_db, f)
        except Exception as e:
            print(f"Warning: Could not save database ({str(e)})")

    def match_features(self, query_features, min_similarity=None):
        if min_similarity is None:
            min_similarity = self.config.feature_match_threshold
            
        max_similarity = 0
        matched_id = None
        
        for obj_id, stored_features in self.feature_db.items():
            for features in stored_features:
                similarity = np.dot(query_features, features)
                if similarity > max_similarity:
                    max_similarity = similarity
                    matched_id = obj_id
                    
        if max_similarity < min_similarity:
            return None, max_similarity
            
        return matched_id, max_similarity

    def update_feature_array(self, obj_id, new_features):
        if obj_id not in self.feature_db:
            self.feature_db[obj_id] = []
            
        should_add = True
        for existing_features in self.feature_db[obj_id]:
            similarity = np.dot(new_features, existing_features)
            if similarity > 0.95:
                should_add = False
                break
                
        if should_add:
            if len(self.feature_db[obj_id]) >= self.config.max_features_per_id:
                self.feature_db[obj_id].pop(0)
            self.feature_db[obj_id].append(new_features)

    def get_next_id(self):
        current_id = self.next_id
        self.next_id += 1
        return current_id