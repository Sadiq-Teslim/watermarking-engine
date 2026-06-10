"""FPWM FastAPI application entrypoint."""
from fastapi import FastAPI

from app.routes import detect, health, watermark

app = FastAPI(
    title="FPWM — FairPlay Watermark Service",
    version="0.1.1",
    description="Self-hosted forensic video watermarking engine.",
)

app.include_router(health.router)
app.include_router(watermark.router)
app.include_router(detect.router)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {"service": "fpwm", "version": app.version}
