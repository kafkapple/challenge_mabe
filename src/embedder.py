from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

@dataclass 
class PCAEmbedder:
    """PCA 기반 프레임 임베딩 클래스"""
    
    def __init__(self, embed_size: int):
        self.embed_size = embed_size
        self.scalers: List[StandardScaler] = []
        self.pca_models: List[PCA] = []
        
    def fit(self, train_data: np.ndarray) -> None:
        """각 마우스 중심 데이터에 대해 PCA 학습"""
        for m in range(3):
            scaler = StandardScaler(with_std=False)
            pca = PCA(n_components=self.embed_size)
            
            scaler.fit(train_data[:,:,m])
            pca.fit(train_data[:,:,m])
            
            self.scalers.append(scaler)
            self.pca_models.append(pca)
            
    def transform(self, keypoints: np.ndarray) -> np.ndarray:
        """키포인트를 임베딩으로 변환"""
        embeddings = np.empty((len(keypoints), self.embed_size*3), dtype=np.float32)
        
        for center_mouse in range(3):
            # Center the keypoints
            ctr = np.median(keypoints[:,center_mouse,:,:], axis=1)
            ctr = np.repeat(np.expand_dims(ctr,axis=1), 3, axis=1)
            ctr = np.repeat(np.expand_dims(ctr,axis=2), 12, axis=2)
            keypoints_centered = keypoints - ctr
            keypoints_centered = keypoints_centered.reshape(keypoints_centered.shape[0], -1)
            
            # Transform
            x = self.scalers[center_mouse].transform(keypoints_centered)
            start_idx = center_mouse * self.embed_size
            end_idx = (center_mouse + 1) * self.embed_size
            embeddings[:,start_idx:end_idx] = self.pca_models[center_mouse].transform(x)
            
        return embeddings 