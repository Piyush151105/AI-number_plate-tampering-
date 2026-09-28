from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    project_name: str = "AI-Enabled Vehicle Number Plate Tampering Detection"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    streamlit_port: int = 8501

    base_dir: Path = Path(__file__).resolve().parent.parent
    upload_dir: Path = base_dir / "uploads"
    output_dir: Path = base_dir / "outputs"
    model_dir: Path = base_dir / "models" / "weights"
    sample_dir: Path = base_dir / "data" / "samples"

    # Indian plate pattern: KA01AB1234, DL8CAD1234, etc.
    plate_regex: str = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"

    tamper_threshold: float = 0.52
    ocr_languages: list[str] = ["en"]
    use_gpu: bool = False

    # e-Challan Provider Configuration
    challan_provider: str = "mock"
    challan_api_base_url: str = ""
    challan_api_key: str = ""
    challan_timeout_sec: int = 8
    challan_rate_limit_per_min: int = 20

    # Tampered Plate Candidate Analysis Configuration
    rto_mapping_path: Path = base_dir / "data" / "rto_mapping.json"
    candidate_min_readable_chars: int = 6
    candidate_max_count: int = 10
    candidate_low_conf_threshold: float = 0.60


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
settings.model_dir.mkdir(parents=True, exist_ok=True)
settings.sample_dir.mkdir(parents=True, exist_ok=True)
