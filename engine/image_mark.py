"""Single-frame watermark: DCT-domain QIM on the luma channel.

Each 8x8 luma block carries ONE codeword bit, embedded by quantizing a mid-frequency
DCT coefficient to an even/odd multiple of Q (QIM). A fixed seeded permutation scatters
each codeword bit across many blocks (spread-spectrum redundancy); extraction majority-votes
the blocks belonging to each bit. This is robust to JPEG/H.264 recompression because it
operates on the same block-DCT structure those codecs use.
"""
import cv2
import numpy as np

from engine.constants import BLOCK, COEF, DEFAULT_Q, SEED


def _block_bit_map(n_blocks: int, n_bits: int) -> np.ndarray:
    """Deterministic mapping: block index -> codeword bit index. Same for embed & extract
    at a given resolution (block count)."""
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_blocks)
    mapping = np.empty(n_blocks, dtype=np.int64)
    mapping[perm] = np.arange(n_blocks) % n_bits
    return mapping


def _qim_level(coef: float, bit: int, q: float) -> int:
    """Nearest quantization level to `coef/q` whose parity equals `bit`."""
    r = coef / q
    k = int(np.floor(r + 0.5))
    if (k & 1) == bit:
        return k
    lower, upper = k - 1, k + 1
    return lower if abs(r - lower) <= abs(r - upper) else upper


def _grid(shape: tuple[int, int]) -> tuple[int, int, int]:
    h, w = shape
    bh, bw = h // BLOCK, w // BLOCK
    return bh, bw, bh * bw


def embed_bits_in_frame(frame_bgr: np.ndarray, bits: list[int], q: float = DEFAULT_Q) -> np.ndarray:
    n_bits = len(bits)
    ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycc[:, :, 0]
    bh, bw, n_blocks = _grid(y.shape)
    bmap = _block_bit_map(n_blocks, n_bits)

    idx = 0
    for by in range(bh):
        for bx in range(bw):
            bit = bits[int(bmap[idx])]
            y0, x0 = by * BLOCK, bx * BLOCK
            block = y[y0:y0 + BLOCK, x0:x0 + BLOCK]
            d = cv2.dct(block)
            d[COEF] = _qim_level(float(d[COEF]), bit, q) * q
            y[y0:y0 + BLOCK, x0:x0 + BLOCK] = cv2.idct(d)
            idx += 1

    ycc[:, :, 0] = np.clip(y, 0, 255)
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def extract_bits_from_frame(frame_bgr: np.ndarray, n_bits: int, q: float = DEFAULT_Q) -> list[int]:
    ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycc[:, :, 0]
    bh, bw, n_blocks = _grid(y.shape)
    bmap = _block_bit_map(n_blocks, n_bits)

    votes0 = np.zeros(n_bits, dtype=np.int64)
    votes1 = np.zeros(n_bits, dtype=np.int64)

    idx = 0
    for by in range(bh):
        for bx in range(bw):
            y0, x0 = by * BLOCK, bx * BLOCK
            d = cv2.dct(y[y0:y0 + BLOCK, x0:x0 + BLOCK])
            k = int(np.floor(float(d[COEF]) / q + 0.5))
            bit_index = int(bmap[idx])
            if k & 1:
                votes1[bit_index] += 1
            else:
                votes0[bit_index] += 1
            idx += 1

    return [1 if votes1[i] >= votes0[i] else 0 for i in range(n_bits)]
