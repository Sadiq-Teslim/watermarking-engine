"""Engine-wide constants. Detector MUST use the same values as the embedder."""

# --- DCT-QIM ---
BLOCK = 8                  # 8x8 DCT blocks (aligned with JPEG/H.264 transform grid)
# Low-frequency coefficient: JPEG/H.264 quantize (2,1) ~4x more gently than mid-band
# (4,3), so the mark survives q35-q90 recompression (validated by bench/tune_qim.py).
COEF = (2, 1)
SEED = 0x46504157          # 'FPAW' — fixed seed for the block->bit permutation
DEFAULT_Q = 20.0           # QIM step: survives JPEG q35+ at PSNR ~43 (tune_qim sweep)

# --- Payload message format (48 bits total) ---
VERSION = 1
VERSION_BITS = 4           # uint4
PAYLOAD_BITS = 28          # uint28  -> up to 268,435,455 ids
CRC_BITS = 16              # uint16
HEADER_BITS = VERSION_BITS + PAYLOAD_BITS   # 32
MESSAGE_BITS = HEADER_BITS + CRC_BITS        # 48
MESSAGE_BYTES = MESSAGE_BITS // 8            # 6

# --- ECC (Reed-Solomon) ---
NSYM = 8                   # parity bytes -> corrects up to NSYM//2 = 4 byte errors
CODEWORD_BYTES = MESSAGE_BYTES + NSYM        # 14
CODEWORD_BITS = CODEWORD_BYTES * 8           # 112

# --- Detection ---
DETECT_SAMPLE_FPS = 1.0    # sample 1 frame/sec from a suspect clip
MIN_CONFIDENCE = 0.6       # winner votes / valid frames
MIN_VALID_FRAMES = 3       # need at least this many CRC-valid frames to accept
CANONICAL_HEIGHTS = [1080, 720, 480, 360]   # best-effort multi-scale resync at detect

# --- Embedding cadence ---
DEFAULT_MARK_STRIDE = 1    # mark every Nth frame (1 = every frame; max redundancy)

MAX_PAYLOAD_ID = (1 << PAYLOAD_BITS) - 1

# --- Audio channel (P6, AudioSeal) ---
AUDIO_SR = 16000           # AudioSeal operates at 16 kHz
AUDIO_NBITS = 16           # AudioSeal 16-bit message capacity
AUDIO_ALPHA = 1.0          # watermark strength
AUDIO_DETECT_THRESHOLD = 0.5

# --- Video neural tier (P7, VideoSeal) ---
NEURAL_DETECT_THRESHOLD = 0.5
