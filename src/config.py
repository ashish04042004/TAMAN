from dataclasses import dataclass


@dataclass
class Config:
    train_csv: str = "data/processed/train.csv"
    val_csv: str = "data/processed/val.csv"
    test_csv: str = "data/processed/test.csv"
    image_root: str = "data/raw"
    target_col: str = "aqi"
    metadata_columns: tuple[str, ...] = (
        "humidity",
        "temperature",
        "hour",
        "is_night",
        "season_code",
        "pm25",
        "pm10",
    )

    # --- TAMAN: set use_taman=False to train/evaluate single-image multimodal instead ---
    use_taman: bool = True
    taman_seq_len: int = 4
    taman_max_consecutive_frame_gap: int = 250
    taman_backbone: str = "resnet18"  # "resnet18" | "resnet50"
    taman_lstm_hidden: int = 256
    taman_lstm_layers: int = 1
    taman_meta_embed_dim: int = 128
    taman_fusion_dim: int = 128
    taman_head_hidden: int = 256
    taman_model_output_path: str = "models/best_taman_r18.pt"
    taman_train_history_output_path: str = "outputs/train_history_taman_r18.json"
    taman_metrics_output_path: str = "outputs/eval/taman_r18/test_metrics.json"
    taman_predictions_output_path: str = "outputs/eval/taman_r18/test_predictions.csv"

    # Single-image multimodal backbone (must match checkpoint when training / evaluating)
    multimodal_backbone: str = "resnet50"

    # TAMAN uses seq_len images per sample; use 4–8 on ~4GB GPUs, higher if memory allows
    batch_size: int = 4
    num_workers: int = 0
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    epochs: int = 10
    random_seed: int = 42
    huber_delta: float = 15.0
    high_aqi_threshold: float = 150.0
    high_aqi_weight: float = 1.6
    use_amp: bool = False

    model_output_path: str = "models/best_multimodal_aqi_v4_resnet50.pt"
    # Default eval artifacts live under outputs/eval/<version>/ (see run_all_evaluations.py)
    metrics_output_path: str = "outputs/eval/v4/test_metrics.json"
    train_history_output_path: str = "outputs/train_history_v4_resnet50.json"
    predictions_output_path: str = "outputs/eval/v4/test_predictions.csv"
