import hydra
from omegaconf import DictConfig
import numpy as np
import os
from visualizer import MouseVisualizer
from trainer import prepare_pca_training_data
from data_handler import MouseTrackingData
from embedder import PCAEmbedder

def seed_everything(seed: int) -> None:
    """시드 설정"""
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set seed
    seed_everything(cfg.seed)
    # Initialize data handler
    data_handler = MouseTrackingData(
        cfg.data.paths.dataset_dir,
        cfg.data.paths.submission_dir, 
        cfg.data.challenge_name
    )
    
    # Download and load data
    data_handler.download_data()
    try:
        submission_clips, user_train = data_handler.load_data()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return
    
    # Initialize PCA embedder
    embedder = PCAEmbedder(cfg.embed_size)
    
    # Prepare training data for PCA
    sequence_keys = list(user_train['sequences'].keys())
    num_total_frames = sum(seq["keypoints"].shape[0] 
                          for _, seq in submission_clips['sequences'].items())
    
    # Train PCA
    pca_train = prepare_pca_training_data(user_train, sequence_keys)
    embedder.fit(pca_train)
    
    # Visualize sample sequence
    visualizer = MouseVisualizer(
        frame_width=cfg.frame_width,
        frame_height=cfg.frame_height,
        mouse_colors=cfg.visualization.mouse_colors,
        plot_pairs=cfg.visualization.plot_pairs
    )
    
    # Sample visualization of first sequence
    sequence_key = list(user_train['sequences'].keys())[0]
    keypoints = user_train['sequences'][sequence_key]['keypoints']
    filled_sequence = MouseTrackingData.fill_holes(keypoints)
    
    visualizer.animate_sequence(
        sequence_key,
        filled_sequence,
        start_frame=0,
        stop_frame=100,  # Visualize first 100 frames
        skip=2,  # Skip every other frame for faster visualization
        cfg=cfg
    )
    # Generate embeddings for submission
    data_handler.generate_submission(submission_clips, embedder, cfg)
   
    
if __name__ == "__main__":
    main() 