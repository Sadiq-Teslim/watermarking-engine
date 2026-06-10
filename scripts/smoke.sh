#!/usr/bin/env bash
# Post-deploy smoke test. Verifies health, auth, and a full embed->status round trip.
#   FPWM_BASE_URL=https://fpwm.example.com FPWM_API_KEY=... TEST_SOURCE_URL=https://.../clip.mp4 \
#     ./scripts/smoke.sh
set -euo pipefail

BASE="${FPWM_BASE_URL:?set FPWM_BASE_URL}"
KEY="${FPWM_API_KEY:?set FPWM_API_KEY}"
AUTH="Authorization: Bearer ${KEY}"

echo "1) /healthz"
curl -fsS "${BASE}/healthz"; echo

echo "2) /readyz"
curl -fsS "${BASE}/readyz"; echo

echo "3) auth rejects missing key (expect 401)"
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE}/v1/watermark/video" \
  -H "Content-Type: application/json" -d '{"source_url":"https://x/y.mp4","payload":1}')
[ "$code" = "401" ] && echo "  ok (401)" || { echo "  FAIL got $code"; exit 1; }

if [ -z "${TEST_SOURCE_URL:-}" ]; then
  echo "TEST_SOURCE_URL not set — skipping embed round trip."
  exit 0
fi

echo "4) submit embed job"
JOB=$(curl -fsS -X POST "${BASE}/v1/watermark/video" -H "$AUTH" \
  -H "Content-Type: application/json" \
  -d "{\"source_url\":\"${TEST_SOURCE_URL}\",\"payload\":12345}")
echo "  $JOB"
JOB_ID=$(echo "$JOB" | sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p')

echo "5) poll status"
for i in $(seq 1 60); do
  S=$(curl -fsS "${BASE}/v1/watermark/jobs/${JOB_ID}" -H "$AUTH")
  STATUS=$(echo "$S" | sed -n 's/.*"status":"\([^"]*\)".*/\1/p')
  echo "  [$i] $STATUS"
  [ "$STATUS" = "ready" ] && { echo "  DONE: $S"; exit 0; }
  [ "$STATUS" = "error" ] && { echo "  FAILED: $S"; exit 1; }
  sleep 5
done
echo "  timed out waiting for job"; exit 1
