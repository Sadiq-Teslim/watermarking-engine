"""Object storage for watermarked output. Default backend: Cloudinary (reuses FairPlay's
account). Returns a stable secure URL that FairPlay stores in Movie.watermarkedUrl."""
import os

import cloudinary
import cloudinary.uploader

from app.config import Settings

# Cloudinary's standard upload handles up to 100MB; use the chunked endpoint above that.
_UPLOAD_LARGE_THRESHOLD = 90 * 1024 * 1024


def _configure(settings: Settings) -> None:
    cloudinary.config(
        cloud_name=settings.cloudinary_cloud_name,
        api_key=settings.cloudinary_api_key,
        api_secret=settings.cloudinary_api_secret,
        secure=True,
    )


def upload_video(settings: Settings, local_path: str, public_id: str | None = None) -> str:
    if settings.storage_backend != "cloudinary":
        raise RuntimeError(f"unsupported storage backend: {settings.storage_backend}")
    if not settings.storage_configured():
        raise RuntimeError("storage backend not configured")
    _configure(settings)
    opts = dict(
        resource_type="video",
        folder=settings.cloudinary_folder,
        public_id=public_id,
        use_filename=False,
        unique_filename=True,
        overwrite=False,
    )
    if os.path.getsize(local_path) > _UPLOAD_LARGE_THRESHOLD:
        result = cloudinary.uploader.upload_large(local_path, chunk_size=20 * 1024 * 1024, **opts)
    else:
        result = cloudinary.uploader.upload(local_path, **opts)
    return result["secure_url"]


def upload_image_bytes(
    settings: Settings,
    data: bytes,
    public_id: str | None = None,
    folder_suffix: str = "images",
) -> dict:
    if settings.storage_backend != "cloudinary":
        raise RuntimeError(f"unsupported storage backend: {settings.storage_backend}")
    if not settings.storage_configured():
        raise RuntimeError("storage backend not configured")
    _configure(settings)
    result = cloudinary.uploader.upload(
        data,
        resource_type="image",
        folder=f"{settings.cloudinary_folder}/{folder_suffix}".strip("/"),
        public_id=public_id,
        use_filename=False,
        unique_filename=True,
        overwrite=False,
    )
    return {
        "url": result["secure_url"],
        "public_id": result.get("public_id", ""),
        "width": result.get("width"),
        "height": result.get("height"),
    }
