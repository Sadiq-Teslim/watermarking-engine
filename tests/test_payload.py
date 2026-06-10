"""Payload encode/decode + checksum-gate tests."""
import pytest

from engine.constants import MAX_PAYLOAD_ID, MESSAGE_BYTES
from engine.payload import decode_message, encode_message


def test_roundtrip_crc():
    for pid in (1, 42, 1_000_000, MAX_PAYLOAD_ID):
        msg = encode_message(pid)
        assert len(msg) == MESSAGE_BYTES
        ok, got, ver = decode_message(msg)
        assert ok and got == pid and ver == 1


def test_corruption_rejected_by_crc():
    msg = bytearray(encode_message(12345))
    msg[0] ^= 0xFF  # flip bits in the header
    ok, _, _ = decode_message(bytes(msg))
    assert ok is False


def test_hmac_mode_roundtrip_and_forgery_rejected():
    secret = b"super-secret"
    msg = encode_message(777, secret=secret)
    ok, got, _ = decode_message(msg, secret=secret)
    assert ok and got == 777
    # Wrong secret must fail the checksum.
    ok2, _, _ = decode_message(msg, secret=b"other-secret")
    assert ok2 is False


def test_payload_bounds():
    with pytest.raises(ValueError):
        encode_message(0)
    with pytest.raises(ValueError):
        encode_message(MAX_PAYLOAD_ID + 1)


def test_decode_wrong_length():
    ok, _, _ = decode_message(b"\x00\x01")
    assert ok is False
