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

    batch_size: int = 12
    num_workers: int = 0
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    epochs: int = 12
    random_seed: int = 42
    huber_delta: float = 15.0
    high_aqi_threshold: float = 150.0
    high_aqi_weight: float = 1.6
    use_amp: bool = False

    model_output_path: str = "models/best_multimodal_aqi_v3.pt"
    metrics_output_path: str = "outputs/test_metrics_v3.json"
    train_history_output_path: str = "outputs/train_history_v3.json"
    predictions_output_path: str = "outputs/test_predictions_v3.csv"
