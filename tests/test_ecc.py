"""Reed-Solomon ECC + bit helper tests."""
import pytest

from engine.constants import NSYM
from engine.ecc import (
    ReedSolomonError,
    bits_to_bytes,
    bytes_to_bits,
    rs_decode,
    rs_encode,
)


def test_bits_roundtrip():
    data = bytes(range(20))
    assert bits_to_bytes(bytes_to_bits(data)) == data


def test_bits_length_validation():
    with pytest.raises(ValueError):
        bits_to_bytes([1, 0, 1])  # not a multiple of 8


def test_rs_corrects_up_to_t_errors():
    data = b"FAIRPLAY"
    cw = bytearray(rs_encode(data, NSYM))
    t = NSYM // 2  # correctable byte errors
    for i in range(t):
        cw[i] ^= 0xFF
    assert rs_decode(bytes(cw), NSYM) == data


def test_rs_fails_beyond_capacity():
    data = b"FAIRPLAY"
    cw = bytearray(rs_encode(data, NSYM))
    for i in range(NSYM):  # exceed t
        cw[i] ^= 0xFF
    with pytest.raises(ReedSolomonError):
        rs_decode(bytes(cw), NSYM)
