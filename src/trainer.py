from data_handler import MouseTrackingData
from embedder import PCAEmbedder
from typing import List, Dict
from tqdm import tqdm
import numpy as np

def prepare_pca_training_data(user_train: Dict, sequence_keys: List[str]) -> np.ndarray:
    """PCA 학습을 위한 데이터 준비"""
    print("\nPreparing PCA training data...")
    sequence_dim = user_train['sequences'][sequence_keys[0]]['keypoints'].shape
    keypoints_dim = sequence_dim[1] * sequence_dim[2] * sequence_dim[3]
    total_frames = sum(len(user_train['sequences'][k]['keypoints']) 
                     for k in sequence_keys)
    print(f"Total frames to process: {total_frames}")
    pca_train = np.empty((total_frames, keypoints_dim, 3), dtype=np.float32)
    
    start = 0
    for k in tqdm(sequence_keys, desc="Processing sequences"):
        keypoints = MouseTrackingData.fill_holes(user_train['sequences'][k]["keypoints"])
        if keypoints.size == 0:
            print(f"Skipping sequence {k} - empty keypoints")
            continue
            
        # Reshape and store in pre-allocated array
        frames = len(keypoints)
        for center_mouse in range(3):
            ctr = np.median(keypoints[:,center_mouse,:,:], axis=1)
            ctr = np.repeat(np.expand_dims(ctr,axis=1), 3, axis=1)
            ctr = np.repeat(np.expand_dims(ctr,axis=2), 12, axis=2)
            keypoints_centered = keypoints - ctr
            pca_train[start:start+frames, :, center_mouse] = keypoints_centered.reshape(frames, -1)
        start += frames
        if start % 1000 == 0:
            print(f"Processed {start}/{total_frames} frames")
    
    print("PCA training data preparation completed")
    return pca_train
