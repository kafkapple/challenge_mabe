from data_handler import MouseTrackingData
from embedder import PCAEmbedder
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed

def process_sequence(k: str, user_train: Dict) -> Tuple[np.ndarray, int]:
    """단일 시퀀스 처리"""
    keypoints = MouseTrackingData.fill_holes(user_train['sequences'][k]["keypoints"])
    if keypoints.size == 0:
        return None, 0
        
    frames = len(keypoints)
    sequence_dim = keypoints.shape
    keypoints_dim = sequence_dim[1] * sequence_dim[2] * sequence_dim[3]
    sequence_data = np.empty((frames, keypoints_dim, 3), dtype=np.float32)
    
    for center_mouse in range(3):
        ctr = np.median(keypoints[:,center_mouse,:,:], axis=1)
        ctr = np.repeat(np.expand_dims(ctr,axis=1), 3, axis=1)
        ctr = np.repeat(np.expand_dims(ctr,axis=2), 12, axis=2)
        keypoints_centered = keypoints - ctr
        sequence_data[:, :, center_mouse] = keypoints_centered.reshape(frames, -1)
        
    return sequence_data, frames

def prepare_embedding_data_parallel(user_train: Dict, sequence_keys: List[str], n_jobs: int = -1) -> np.ndarray:
    """임베딩을 위한 데이터 준비 (병렬 처리)"""
    print("\nPreparing embedding training data...")
    
    # 병렬 처리로 시퀀스 처리
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_sequence)(k, user_train) for k in tqdm(sequence_keys)
    )
    
    # 유효한 결과만 필터링
    valid_results = [(data, frames) for data, frames in results if data is not None]
    
    # 결과 합치기
    total_frames = sum(frames for _, frames in valid_results)
    sequence_dim = user_train['sequences'][sequence_keys[0]]['keypoints'].shape
    keypoints_dim = sequence_dim[1] * sequence_dim[2] * sequence_dim[3]
    
    training_data = np.empty((total_frames, keypoints_dim, 3), dtype=np.float32)
    
    start = 0
    for data, frames in valid_results:
        training_data[start:start+frames] = data
        start += frames
        
    print("Embedding data preparation completed")
    return training_data
