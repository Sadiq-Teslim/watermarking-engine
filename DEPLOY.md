# FPWM Deployment Runbook

FPWM is the self-hosted FairPlay Watermark Service. It deploys independently of the FairPlay
app; FairPlay talks to it through the `fairplay` watermark provider over HTTPS.

## Pre-flight (do NOT skip — this is the accuracy gate)

1. CI must be green, including the **benchmark gate**:
   ```
   python -m bench.run_benchmark && python -m bench.gates
   ```
   Do not deploy a build whose accuracy gates fail. (See `bench/gates.py`.)
2. Decide secrets (generate once, store in your secret manager):
   - `FPWM_API_KEY` — shared with the FairPlay server (`FPWM_API_KEY` there).
   - `FPWM_HMAC_SECRET` — payload integrity + signed callbacks.
   - Cloudinary creds — same account as FairPlay (`CLOUDINARY_*`).
   - Managed Redis with `maxmemory-policy noeviction` (job results must not be evicted).

## Option A — Render (CPU host, simplest)

1. Push the repo; in Render, "New > Blueprint" and select `watermark-engine/render.yaml`.
2. After first deploy, set the `sync: false` secrets in the dashboard:
   `FPWM_API_KEY`, `FPWM_HMAC_SECRET`, `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`,
   `CLOUDINARY_API_SECRET`.
3. Redis is provisioned by the blueprint and wired via `REDIS_URL` automatically.
4. Scale `fpwm-worker` (`numInstances`) for throughput; the `web` service stays at 1–2.

## Option B — Any Docker host (VM / ECS / Fly)

1. Copy `.env.production.example` to `.env` on the host and fill it (or inject via the
   platform's env/secret mechanism).
2. Build + run:
   ```
   docker compose -f docker-compose.prod.yml up -d --build --scale worker=2
   ```
3. Put the `web` service behind TLS (reverse proxy / platform LB). Keep Redis private.

## Wire FairPlay to FPWM

On the FairPlay server environment:
```
WATERMARK_PROVIDER=fairplay
FPWM_BASE_URL=https://<your-fpwm-host>
FPWM_API_KEY=<same key as FPWM>
WATERMARK_VERIFY_ON_SCAN=true   # optional: forensic-verify found copies (bandwidth-heavy)
```
Then restart the FairPlay API and worker. `validateContentProtectionConfig()` enforces that
`FPWM_BASE_URL` and `FPWM_API_KEY` are present in production.

## Smoke test

```
FPWM_BASE_URL=https://<host> FPWM_API_KEY=<key> \
TEST_SOURCE_URL=https://<a-public-test-clip>.mp4 \
  ./scripts/smoke.sh
```
Checks `/healthz`, `/readyz`, auth rejection, and a full embed→ready round trip.

## Enabling the neural tier (audio + VideoSeal)

The neural tier is opt-in (heavy torch deps). To enable:

1. Build the image with neural deps:
   ```
   INSTALL_NEURAL=true docker compose -f docker-compose.prod.yml up -d --build
   ```
   (Render: add `INSTALL_NEURAL=true` as a build env var.)
2. Validate accuracy for each neural engine BEFORE switching production traffic:
   ```
   AUDIO_WATERMARK_ENABLED=true FPWM_BENCH_ENGINE=videoseal \
     python -m bench.run_benchmark && python -m bench.gates
   ```
   The `videoseal` run additionally enforces the geometric gates (resize/crop/rotate).
3. Turn it on:
   - Audio channel: set `AUDIO_WATERMARK_ENABLED=true` on FPWM.
   - Video neural engine: set `FPWM_ENGINE=videoseal` on the FairPlay server.
4. GPU: for throughput, build torch with CUDA (swap the index URL in
   `requirements-neural.txt`) and run workers on a GPU host. CPU works but is slower.

## Operate

- **Health:** `/healthz` (liveness), `/readyz` (redis + ffmpeg + storage).
- **Scaling:** add worker instances; cap per-worker concurrency (ffmpeg is RAM-heavy — one
  job per worker process by design).
- **Rollback:** flip `WATERMARK_PROVIDER` back to `imatag` on FairPlay; the abstraction
  makes FPWM hot-swappable.
- **Long jobs:** embedding is always async; HTTP requests never block on transcode. Tune
  `MAX_DURATION_S` to bound runaway inputs.

## Security checklist

- TLS in front of `web`; Redis never publicly exposed.
- `FPWM_API_KEY` rotated via secret manager (update both FPWM and FairPlay together).
- SSRF guard active on `source_url` (blocks private/loopback targets).
- Cloudinary creds scoped to the watermarking account.
