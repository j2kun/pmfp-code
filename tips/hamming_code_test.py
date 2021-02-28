from assertpy import assert_that
from bitstring import BitStream
from hypothesis import given
from hypothesis.strategies import binary
from hypothesis.strategies import composite
from hypothesis.strategies import integers

from hamming_code import decode
from hamming_code import encode


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
