import hydra
from omegaconf import DictConfig
import numpy as np
import os
from visualizer import MouseVisualizer
from trainer import prepare_embedding_data_parallel
from data_handler import MouseTrackingData
from embedder import PCAEmbedder, get_embedder
from omegaconf import OmegaConf
import wandb
import pytorch_lightning as pl
from evaluation import EvaluationMetrics
import time
from pathlib import Path


def seed_everything(seed: int) -> None:
    """시드 설정"""
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    
    selected_config = OmegaConf.select(cfg, 'params')
    run_name = cfg.wandb.name if 'name' in cfg.wandb else f"run_{int(time.time())}"
    
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        config=OmegaConf.to_container(selected_config, resolve=True)
    )

    pl.seed_everything(cfg.general.seed)
    # Set seed
    #seed_everything(cfg.general.seed)
    # Initialize data handler
    base_dir = Path(cfg.data.paths.base_dir)
    vis_dir = base_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    data_handler = MouseTrackingData(
        str(Path(cfg.data.paths.dataset_dir)),
        str(Path(cfg.data.paths.submission_dir)), 
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
    embedder = PCAEmbedder(cfg.params.embed_size)
    
    # Prepare training data for PCA
    sequence_keys = list(user_train['sequences'].keys())
    num_total_frames = sum(seq["keypoints"].shape[0] 
                          for _, seq in submission_clips['sequences'].items())
    
    # 데이터 정보 출력
    if cfg.params.evaluation.verbose:
        EvaluationMetrics.print_data_info(user_train, "Training Data")
        EvaluationMetrics.print_data_info(submission_clips, "Submission Data")
    
    # 학습/검증 데이터 분할
    train_data, val_data = EvaluationMetrics.split_train_val(
        user_train,
        train_ratio=cfg.params.evaluation.train_ratio,
        seed=cfg.general.seed
    )
    
    # 학습 데이터로 임베더 학습
    sequence_keys = list(train_data['sequences'].keys())
    
    # 샘플링 적용
    if cfg.params.embedding.sampling_ratio < 1.0:
        num_sequences = len(sequence_keys)
        sample_size = int(num_sequences * cfg.params.embedding.sampling_ratio)
        sequence_keys = np.random.choice(sequence_keys, size=sample_size, replace=False)
    
    # 임베딩 학습 데이터 준비 (병렬 처리)
    train_data = prepare_embedding_data_parallel(
        train_data, 
        sequence_keys,
        n_jobs=cfg.params.embedding.n_jobs
    )
    
    # 임베더 학습
    embedder = get_embedder(cfg.params.embedding.method, cfg.params.embed_size)
    embedder.fit(train_data)
    
    # 검증 데이터에 대한 임베딩 생성
    val_embeddings = []
    for seq_key in val_data['sequences']:
        keypoints = val_data['sequences'][seq_key]['keypoints']
        val_embeddings.append(embedder.transform(keypoints))
    val_embeddings = np.concatenate(val_embeddings)
    
    # 메트릭 계산 (예시)
    if cfg.params.evaluation.metrics:
        # 여기서는 같은 임베딩끼리 비교 (실제로는 ground truth와 비교해야 함)
        metrics = EvaluationMetrics.calculate_metrics(
            val_embeddings[:len(val_embeddings)//2],
            val_embeddings[len(val_embeddings)//2:],
            cfg.params.evaluation.metrics
        )
        
        print("\nValidation Metrics:")
        for metric_name, value in metrics.items():
            print(f"- {metric_name}: {value:.4f}")
            wandb.log({f"val_{metric_name}": value})
    
    # Visualize sample sequence
    visualizer = MouseVisualizer(
        frame_width=cfg.params.frame_width,
        frame_height=cfg.params.frame_height,
        mouse_colors=cfg.visualization.mouse_colors,
        plot_pairs=cfg.visualization.plot_pairs
    )
    
    # Sample visualization of first sequence
    sequence_key = list(user_train['sequences'].keys())[0]
    keypoints = user_train['sequences'][sequence_key]['keypoints']
    filled_sequence = MouseTrackingData.fill_holes(keypoints)
    
    wandb.finish()
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