"""FPWM FastAPI application entrypoint."""
from fastapi import FastAPI

from app.routes import detect, health, image, watermark

app = FastAPI(
    title="FPWM — FairPlay Watermark Service",
    version="0.2.0",
    description="Self-hosted forensic video + image watermarking engine.",
)

app.include_router(health.router)
app.include_router(watermark.router)
app.include_router(detect.router)
app.include_router(image.router)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {"service": "fpwm", "version": app.version}
