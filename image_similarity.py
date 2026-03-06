import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.cluster import DBSCAN
import numpy as np
from typing import List, Dict, Tuple


class ImageSimilarityDetector:
    def __init__(self, model_name: str = "facebook/dino-vitb16"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def extract_features(self, image_path: str) -> np.ndarray | None:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state[:, 0].cpu().numpy()
            
            return features.flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        return float(np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2)))
    
    def group_similar_images(self, image_paths: List[str], similarity_threshold: float = 0.85, min_samples: int = 2) -> Dict[int, List[str]]:
        print("Extracting features from images...")
        features_dict = {}
        
        for img_path in image_paths:
            features = self.extract_features(img_path)
            if features is not None:
                features_dict[img_path] = features
        
        if len(features_dict) < 2:
            return {0: list(features_dict.keys())}
        
        features_list = list(features_dict.values())
        paths_list = list(features_dict.keys())
        
        features_array = np.array(features_list)
        features_normalized = features_array / np.linalg.norm(features_array, axis=1, keepdims=True)
        
        distance_matrix = 1 - np.dot(features_normalized, features_normalized.T)
        
        clustering = DBSCAN(
            eps=1-similarity_threshold,
            min_samples=min_samples,
            metric="precomputed"
        )
        labels = clustering.fit_predict(distance_matrix)
        
        groups = {}
        for idx, label in enumerate(labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(paths_list[idx])
        
        return groups


def test_detector():
    detector = ImageSimilarityDetector()
    print("Image Similarity Detector initialized successfully!")
    print(f"Using device: {detector.device}")


if __name__ == "__main__":
    test_detector()
