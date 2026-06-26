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
4. Scale `fpwm-worker` (`numInstances`) for throughput; keep the `web` service at
   `WEB_CONCURRENCY=1` when TrustMark is installed so the neural model is not duplicated
   across multiple web processes on a CPU host.

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

## Enabling the image neural tier (TrustMark / ProofMark Strong mode)

The image neural tier is opt-in because it installs torch + TrustMark. This is what
ProofMark uses for Strong image protection.

1. Build the image with neural deps:
   ```
   INSTALL_NEURAL=true docker compose -f docker-compose.prod.yml up -d --build
   ```
   (Render: add `INSTALL_NEURAL=true` as a build env/build arg.)
2. Confirm the deployed engine reports TrustMark availability:
   ```
   curl -H "Authorization: Bearer <FPWM_API_KEY>" \
     https://<host>/v1/image/capabilities
   ```
   `engines.trustmark.available` remains `false` until `FPWM_TRUSTMARK_ENABLED=true`
   is set on FPWM.
3. Configure the CPU-safe TrustMark runtime:
   ```
   WEB_CONCURRENCY=1
   FPWM_TRUSTMARK_MODEL=C
   FPWM_TRUSTMARK_MAX_SIDE=768
   ```
   `FPWM_TRUSTMARK_MODEL` must stay the same for encode and detect. Increase
   `FPWM_TRUSTMARK_MAX_SIDE` only after memory and latency are measured on the host.
4. Validate image accuracy BEFORE enabling Strong mode for users:
   ```
   FPWM_BENCH_ENGINE=trustmark python -m bench.run_image_benchmark \
     && python -m bench.image_gates
   ```
5. Run a live embed+detect smoke test against the deployed host. Only then set
   `FPWM_TRUSTMARK_ENABLED=true` and redeploy/restart FPWM. ProofMark's Strong mode
   reads `/v1/image/capabilities`, so it stays disabled until this flag is on.

## Enabling the full neural tier (audio + VideoSeal)

The full neural tier includes the image tier plus audio/video models. It is heavier and should
only be enabled when FairPlay video/audio neural watermarking is ready.

1. Build the image with full neural deps:
   ```
   INSTALL_NEURAL_FULL=true docker compose -f docker-compose.prod.yml up -d --build
   ```
   (Render: add `INSTALL_NEURAL_FULL=true` as a build env/build arg.)
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
- **512MB/free alpha mode:** set `FPWM_QUALITY_METRICS_ENABLED=false`,
  `FPWM_X264_PRESET=ultrafast`, and `FPWM_VIDEO_CRF=23`. This keeps the forensic embed path
  active but skips expensive PSNR/SSIM/VMAF analysis and uses a lighter encoder profile.
  Re-enable metrics before production benchmarking.
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
