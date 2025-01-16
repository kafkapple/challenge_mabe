from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from typing import Dict, List, Tuple

def get_embedder(method: str, embed_size: int):
    """임베더 팩토리 함수"""
    if method == "pca":
        return PCAEmbedder(embed_size)
    elif method == "random_projection":
        return RandomProjectionEmbedder(embed_size)
    else:
        raise ValueError(f"Unknown embedding method: {method}")

@dataclass
class BaseEmbedder:
    """기본 임베더 클래스"""
    embed_size: int

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
            
            # Transform (구현은 하위 클래스에서)
            x = self._transform_single(keypoints_centered, center_mouse)
            start_idx = center_mouse * self.embed_size
            end_idx = (center_mouse + 1) * self.embed_size
            embeddings[:,start_idx:end_idx] = x
            
        return embeddings

@dataclass 
class PCAEmbedder(BaseEmbedder):
    """PCA 기반 프레임 임베딩 클래스"""
    
    def __init__(self, embed_size: int):
        super().__init__(embed_size)
        self.scalers: List[StandardScaler] = []
        self.pca_models: List[PCA] = []
        
    def fit(self, train_data: np.ndarray) -> None:
        for m in range(3):
            scaler = StandardScaler(with_std=False)
            pca = PCA(n_components=self.embed_size)
            
            scaler.fit(train_data[:,:,m])
            pca.fit(train_data[:,:,m])
            
            self.scalers.append(scaler)
            self.pca_models.append(pca)
            
    def _transform_single(self, data: np.ndarray, idx: int) -> np.ndarray:
        x = self.scalers[idx].transform(data)
        return self.pca_models[idx].transform(x)

@dataclass 
class RandomProjectionEmbedder(BaseEmbedder):
    """랜덤 프로젝션 기반 프레임 임베딩 클래스"""
    
    def __init__(self, embed_size: int):
        super().__init__(embed_size)
        self.projectors = []
        
    def fit(self, train_data: np.ndarray) -> None:
        for m in range(3):
            projector = GaussianRandomProjection(n_components=self.embed_size)
            projector.fit(train_data[:,:,m])
            self.projectors.append(projector)
            
    def _transform_single(self, data: np.ndarray, idx: int) -> np.ndarray:
        return self.projectors[idx].transform(data) 