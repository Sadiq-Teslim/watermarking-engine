"""Reed-Solomon ECC + bit/byte helpers."""
from reedsolo import ReedSolomonError, RSCodec  # noqa: F401  (re-exported for callers)

from engine.constants import NSYM


def rs_encode(data: bytes, nsym: int = NSYM) -> bytes:
    return bytes(RSCodec(nsym).encode(bytearray(data)))


def rs_decode(data: bytes, nsym: int = NSYM) -> bytes:
    """Return the corrected message bytes (without parity). Raises ReedSolomonError
    if the data has more errors than can be corrected."""
    decoded = RSCodec(nsym).decode(bytearray(data))
    return bytes(decoded[0])


def bytes_to_bits(data: bytes) -> list[int]:
    return [(byte >> (7 - i)) & 1 for byte in data for i in range(8)]


def bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("bit length must be a multiple of 8")
    out = bytearray(len(bits) // 8)
    for i, bit in enumerate(bits):
        if bit:
            out[i // 8] |= 1 << (7 - (i % 8))
    return bytes(out)
