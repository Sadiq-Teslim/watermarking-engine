"""Payload message: version(4) | payload_id(28) | checksum(16).

The checksum is the gate that makes false positives ~0: a payload is only ever returned
when the checksum validates after ECC correction. Two checksum modes:
  - CRC-16-CCITT (default): integrity only.
  - Truncated HMAC-SHA256 (when a secret is supplied): integrity + anti-forgery.
"""
import hashlib
import hmac

from engine.constants import CRC_BITS, MAX_PAYLOAD_ID, MESSAGE_BYTES, PAYLOAD_BITS, VERSION

_CRC_MASK = (1 << CRC_BITS) - 1
_PAYLOAD_MASK = MAX_PAYLOAD_ID


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def _checksum(header_bytes: bytes, secret: bytes | None) -> int:
    if secret:
        digest = hmac.new(secret, header_bytes, hashlib.sha256).digest()
        return int.from_bytes(digest[:2], "big") & _CRC_MASK
    return crc16_ccitt(header_bytes)


def encode_message(payload_id: int, secret: bytes | None = None, version: int = VERSION) -> bytes:
    if not (1 <= payload_id <= MAX_PAYLOAD_ID):
        raise ValueError(f"payload_id must be in [1, {MAX_PAYLOAD_ID}]")
    header = (version << PAYLOAD_BITS) | payload_id          # 32 bits
    header_bytes = header.to_bytes(4, "big")
    chk = _checksum(header_bytes, secret)
    message = (header << CRC_BITS) | chk                     # 48 bits
    return message.to_bytes(MESSAGE_BYTES, "big")


def decode_message(msg_bytes: bytes, secret: bytes | None = None) -> tuple[bool, int, int]:
    """Return (checksum_ok, payload_id, version)."""
    if len(msg_bytes) != MESSAGE_BYTES:
        return (False, 0, 0)
    message = int.from_bytes(msg_bytes, "big")
    chk = message & _CRC_MASK
    header = message >> CRC_BITS
    expected = _checksum(header.to_bytes(4, "big"), secret)
    ok = hmac.compare_digest(chk.to_bytes(2, "big"), expected.to_bytes(2, "big"))
    version = header >> PAYLOAD_BITS
    payload_id = header & _PAYLOAD_MASK
    return (ok, payload_id, version)
