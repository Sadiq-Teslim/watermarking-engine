"""Environment-validated settings for FPWM."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # Auth
    fpwm_api_key: str = ""
    fpwm_hmac_secret: str = ""

    # Infra
    redis_url: str = "redis://localhost:6379/0"

    # Storage
    storage_backend: str = "cloudinary"
    cloudinary_cloud_name: str = ""
    cloudinary_api_key: str = ""
    cloudinary_api_secret: str = ""
    cloudinary_folder: str = "fairplayafrica/watermarked"

    # Job guards
    max_source_bytes: int = 2_000_000_000
    max_duration_s: int = 14_400
    fpwm_quality_metrics_enabled: bool = True
    fpwm_video_crf: int = 18
    fpwm_x264_preset: str = "medium"

    # Neural tier
    audio_watermark_enabled: bool = False
    audio_alpha: float = 1.0
    fpwm_trustmark_enabled: bool = False
    fpwm_trustmark_model: str = "C"
    fpwm_trustmark_max_side: int = 768

    def storage_configured(self) -> bool:
        if self.storage_backend == "cloudinary":
            return bool(
                self.cloudinary_cloud_name
                and self.cloudinary_api_key
                and self.cloudinary_api_secret
            )
        return False


@lru_cache
def get_settings() -> Settings:
    return Settings()
