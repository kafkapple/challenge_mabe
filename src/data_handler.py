from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple, Any
import os
from embedder import PCAEmbedder
from omegaconf import DictConfig
from tqdm import tqdm
from pathlib import Path

@dataclass
class MouseTrackingData:
    """마우스 트래킹 데이터를 처리하는 클래스"""
    dataset_dir: Path
    submission_dir: Path
    challenge_name: str

    def __init__(self, dataset_dir: str, submission_dir: str, challenge_name: str):
        self.dataset_dir = Path(dataset_dir)
        self.submission_dir = Path(submission_dir)
        self.challenge_name = challenge_name
        
        # 디렉토리 생성
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.submission_dir.mkdir(parents=True, exist_ok=True)

    def download_data(self) -> None:
        """AIcrowd에서 데이터 다운로드"""
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)
        
        print("Please check if dataset files exist in:")
        print(f"Dataset directory: {self.dataset_dir}")
        print("\nRequired files:")
        print(f"1. submission_data.npy -> {self.dataset_dir}/")
        print(f"2. user_train.npy -> {self.dataset_dir}/")
        
    def load_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """데이터 로드"""
        submission_path = f'{self.dataset_dir}/submission_data.npy'
        train_path = f'{self.dataset_dir}/user_train.npy'
        
        if not os.path.exists(submission_path) or not os.path.exists(train_path):
            raise FileNotFoundError(
                "\nRequired dataset files not found!"
                "\nPlease download the following files and place them in the dataset directory:"
                f"\n1. submission_data.npy -> {submission_path}"
                f"\n2. user_train.npy -> {train_path}"
                "\n\nYou can download these files from the AIcrowd challenge page."
            )
        
        submission_clips = np.load(submission_path, 
                                 allow_pickle=True).item()
        user_train = np.load(train_path, 
                           allow_pickle=True).item()
        return submission_clips, user_train

    @staticmethod
    def fill_holes(data: np.ndarray) -> np.ndarray:
        """키포인트 데이터의 빈 값(0) 보간"""
        clean_data = data.copy()
        
        # First frame initialization
        for m in range(3):
            holes = np.where(clean_data[0,m,:,0]==0)[0]
            for h in holes:
                sub = np.where(clean_data[:,m,h,0]!=0)[0]
                if len(sub) > 0:
                    clean_data[0,m,h,:] = clean_data[sub[0],m,h,:]
                else:
                    return np.empty((0))
                    
        # Fill remaining frames
        for fr in range(1, clean_data.shape[0]):
            for m in range(3):
                holes = np.where(clean_data[fr,m,:,0]==0)[0]
                for h in holes:
                    clean_data[fr,m,h,:] = clean_data[fr-1,m,h,:]
                    
        return clean_data
    @staticmethod
    def generate_submission(submission_clips: Dict, embedder: PCAEmbedder, cfg: DictConfig) -> Dict:
        """제출용 임베딩 생성"""
        print("\nGenerating submission embeddings...")
        num_total_frames = sum(seq["keypoints"].shape[0] 
                            for _, seq in submission_clips['sequences'].items())
        print(f"Total frames to embed: {num_total_frames}")
        embeddings_array = np.empty((num_total_frames, embedder.embed_size*3), dtype=np.float32)
        
        frame_number_map = {}
        start = 0
        for sequence_key in tqdm(submission_clips['sequences'], desc="Processing submissions"):
            keypoints = MouseTrackingData.fill_holes(submission_clips['sequences'][sequence_key]["keypoints"])
            if keypoints.size == 0:
                #print(f"Using original keypoints for sequence {sequence_key}")
                keypoints = submission_clips['sequences'][sequence_key]["keypoints"]
            
            embeddings = embedder.transform(keypoints)
            end = start + len(keypoints)
            embeddings_array[start:end] = embeddings
            frame_number_map[sequence_key] = (start, end)
            start = end
            # if start % 1000 == 0:
            #     print(f"Embedded {start}/{num_total_frames} frames")

        submission_dict = {"frame_number_map": frame_number_map, "embeddings": embeddings_array}
        
        # Path 객체 사용 및 경로 수정
        submission_dir = Path(cfg.data.paths.submission_dir)
        submission_dir.mkdir(parents=True, exist_ok=True)
        submission_path = submission_dir / "submission.npy"
        
        np.save(str(submission_path), submission_dict)
        print(f"\nSubmission saved to: {submission_path}")
        
        return submission_dict
    