from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split

@dataclass
class EvaluationMetrics:
    """평가 메트릭 계산 클래스"""
    
    @staticmethod
    def print_data_info(data: Dict, name: str = "Dataset") -> None:
        """데이터셋 정보 출력"""
        print(f"\n{name} Information:")
        total_sequences = len(data['sequences'])
        
        # 첫 번째 시퀀스의 키포인트 shape 가져오기
        first_seq = next(iter(data['sequences'].values()))
        keypoint_shape = first_seq['keypoints'].shape
        
        # 전체 프레임 수 계산
        total_frames = sum(seq['keypoints'].shape[0] 
                         for seq in data['sequences'].values())
        
        # 메모리 사용량 계산 (모든 시퀀스의 keypoints 합)
        total_memory = sum(seq['keypoints'].nbytes 
                         for seq in data['sequences'].values())
        
        print(f"- Total sequences: {total_sequences}")
        print(f"- Total frames: {total_frames}")
        print(f"- Keypoint shape: {keypoint_shape}")
        print(f"- Memory usage: {total_memory / 1e9:.2f} GB")
        
    @staticmethod
    def split_train_val(data: Dict, train_ratio: float = 0.8, seed: int = 42) -> Tuple[Dict, Dict]:
        """학습/검증 데이터 분할"""
        sequence_keys = list(data['sequences'].keys())
        train_keys, val_keys = train_test_split(
            sequence_keys, 
            train_size=train_ratio,
            random_state=seed
        )
        
        train_data = {
            'sequences': {k: data['sequences'][k] for k in train_keys}
        }
        val_data = {
            'sequences': {k: data['sequences'][k] for k in val_keys}
        }
        
        return train_data, val_data
        
    @staticmethod
    def calculate_metrics(embeddings1: np.ndarray, 
                         embeddings2: np.ndarray,
                         metrics: List[str]) -> Dict[str, float]:
        """평가 메트릭 계산"""
        results = {}
        
        if "mse" in metrics:
            mse = mean_squared_error(embeddings1, embeddings2)
            results["mse"] = mse
            
        if "f1_score" in metrics:
            # 임베딩을 이진 레이블로 변환 (예시)
            labels1 = (embeddings1 > embeddings1.mean()).astype(int)
            labels2 = (embeddings2 > embeddings2.mean()).astype(int)
            f1 = f1_score(labels1, labels2, average='macro')
            results["f1_score"] = f1
            
        return results 