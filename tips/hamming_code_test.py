from assertpy import assert_that
from bitstring import BitStream
from hypothesis import given
from hypothesis.strategies import binary, composite, integers

from tips.hamming_code import decode, encode


@composite
def bytes_and_index(draw):
    data = draw(binary(min_size=1, max_size=None))
    i = draw(integers(min_value=0, max_value=len(data) * 8 - 1))
    return (data, i)


@given(binary(min_size=0, max_size=None))
def test_encode_decode_is_identity(input_bytes):
    encoded = encode(BitStream(input_bytes))
    assert_that(decode(encoded)).is_equal_to(input_bytes)


@given(bytes_and_index())
def test_can_recover_from_single_bit_error(data_index):
    input_bytes, index = data_index
    encoded = encode(BitStream(input_bytes))
    encoded.invert(index)
    assert_that(decode(encoded)).is_equal_to(input_bytes)


def test_encode_single_block():
    bits = BitStream(bin="0b1011")
    expected = BitStream(bin="0b1011010")
    assert_that(encode(bits)).is_equal_to(expected)


def assert_decodes_with_error_in_bit(bits, encoded, index):
    encoded.pos = 0
    encoded.invert(index)
    assert_that(decode(encoded)).is_equal_to(bits)
    encoded.invert(index)


def test_decode_single_block_with_message_bit_error():
    bits = BitStream(bin="0b1011")
    encoded = encode(bits)
    assert_decodes_with_error_in_bit(bits, encoded, 0)
    assert_decodes_with_error_in_bit(bits, encoded, 1)
    assert_decodes_with_error_in_bit(bits, encoded, 2)
    assert_decodes_with_error_in_bit(bits, encoded, 3)


def test_decode_single_block_with_check_bit_error():
    bits = BitStream(bin="0b1011")
    encoded = encode(bits)
    assert_decodes_with_error_in_bit(bits, encoded, 4)
    assert_decodes_with_error_in_bit(bits, encoded, 5)
    assert_decodes_with_error_in_bit(bits, encoded, 6)
