# FPWM — FairPlay Watermark Service

Self-hosted forensic video watermarking microservice. Embeds an imperceptible,
error-corrected, CRC-gated payload into video (audio + neural tiers added in later batches)
and recovers it from degraded copies. Called by FairPlay through the watermark provider
abstraction.

> Design rationale and accuracy contract live in the repo root: `WATERMARKING_SYSTEM.md`.

## Run locally (Docker)

```bash
cp .env.example .env          # fill FPWM_API_KEY + Cloudinary creds
docker compose up --build     # starts redis + web (:8000) + worker
curl localhost:8000/healthz   # -> {"status":"ok"}
curl localhost:8000/readyz    # -> components: redis/ffmpeg/storage
```

## Run tests (inside the image — real ffmpeg + redis)

```bash
docker compose up -d redis
docker compose run --rm web pytest
docker compose run --rm web ruff check .
```

## Layout

- `app/` — FastAPI app, config, auth, routes, health checks.
- `engine/` — pure watermark core (payload/ECC/embed/extract/voting). *(added P1)*
- `worker/` — RQ worker + tasks. *(tasks added P2)*
- `bench/` — robustness benchmark + acceptance gates. *(added P1)*
- `tests/` — unit + integration tests.

## Status

- [x] P0 — skeleton, Docker, health, auth, tests
- [x] P1 — accuracy core (DCT-QIM + RS-ECC + CRC gate + multi-frame voting) + benchmark
- [x] P2 — watermark/detect API + RQ jobs + Cloudinary storage
- [x] P3 — FairPlay integration (`fairplayProvider.js`, unique payload counter, poll job)
- [x] P4 — detection wired into scan/monitoring (forensic proof on found copies + DMCA)
- [x] P5 — deploy (prod compose, Render blueprint, smoke test, runbook)
- [x] P6 — audio watermark (AudioSeal) — independent 2nd channel (opt-in, INSTALL_NEURAL)
- [x] P7 — neural video tier (VideoSeal) — geometric/screen-record robustness (opt-in)

> Not yet verified end-to-end: this machine cannot run Docker or the ML stack. Verification
> (pytest + ruff + the benchmark gates) runs in CI / at prod deploy per the owner's decision.
