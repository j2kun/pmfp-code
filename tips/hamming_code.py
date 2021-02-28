from bitstring import BitStream
from bitstring import Bits
import numpy


P = numpy.array([
    1, 1, 0,
    1, 0, 1,
    0, 1, 1,
    1, 1, 1,
], dtype=numpy.uint8).reshape((4, 3))

generator_matrix = numpy.concatenate(
    (numpy.identity(n=4, dtype=numpy.uint8), P),
    axis=1
)

parity_check_matrix = numpy.concatenate(
    (P.transpose(), numpy.identity(n=3, dtype=numpy.uint8)),
    axis=1
)

# trick to make parity check exactly
# describe the bit position of the error
permutation = [4, 5, 0, 6, 1, 2, 3]
generator_matrix = generator_matrix[:, permutation]
parity_check_matrix = parity_check_matrix[:, permutation]


syndrome_table = {
  1:4,
  2:2,
  3:6,
  4:1,
  5:5,
  6:3,
  7:7,
}


def encode(bits: BitStream) -> BitStream:
    '''Encode a message to be resilient to errors.

    Arguments:
      - bits: a bit stream to encode

    Returns:
      A bit stream containing the encoded message
    '''
    # convert message into blocks of 4 bits
    num_blocks = len(bits) // 4
    blocks = numpy.zeros((num_blocks, 1), dtype=numpy.uint8)
    for i in range(num_blocks):
        blocks[i] = bits.read('uint:4')
    blocks = numpy.unpackbits(blocks, axis=1)[:, 4:]

    encoded_blocks = blocks.dot(generator_matrix) % 2

    # convert encoded blocks back into bytes
    encoded_blocks = numpy.packbits(
        numpy.concatenate(
            # 7-bit encoded means we need one extra zero in front
            (numpy.zeros((len(encoded_blocks), 1), dtype=numpy.uint8),
                encoded_blocks),
            axis=1
        )
    )
    s = BitStream()
    for block in encoded_blocks:
        s.append(Bits(uint=block, length=7))

    s.pos = 0
    return s


def decode(bits: BitStream) -> BitStream:
    '''Decode a message while also correcting any errors found.

    If the number of errors is too large, the decoded message may be incorrect.

    Arguments:
      - bits: the encoded stream of bits

    Returns:
      A BitStream representing the decoded message
    '''
    num_blocks = len(bits) // 7
    blocks = numpy.zeros((num_blocks, 1), dtype=numpy.uint8)
    for i in range(num_blocks):
        blocks[i] = bits.read('uint:7')
    blocks = numpy.unpackbits(blocks, axis=1)[:, 1:]

    syndromes = parity_check_matrix.dot(blocks.transpose()).transpose() % 2

    # convert to syndromes as integers for each block
    syndromes = numpy.packbits(
        numpy.concatenate(
            (numpy.zeros((num_blocks, 5), dtype=numpy.uint8), syndromes),
            axis=1),
        axis=1).reshape((num_blocks,))

    # correct errors
    for row, syndrome in enumerate(syndromes):
        # if bit==0, there is no error, otherwise error location
        # can be looked up according to syndrome table
        if syndrome:
            bit = syndrome_table[syndrome]
            blocks[(row, bit - 1)] = 1 - blocks[(row, bit - 1)]

    # drop parity check bits and recombine
    decoded_blocks = blocks[:, [2, 4, 5, 6]]
    decoded_blocks = numpy.packbits(
        numpy.concatenate(
            # 4-bit decoded means we need 4 extra zeros in front
            (numpy.zeros((len(decoded_blocks), 4), dtype=numpy.uint8),
                decoded_blocks),
            axis=1
        )
    )
    s = BitStream()
    for block in decoded_blocks:
        s.append(Bits(uint=block, length=4))
    s.pos = 0

    return s


if __name__ == "__main__":
    msg = 'wat'
    print(f'message: {msg}')
    encoded = encode(BitStream(bytes(msg, 'utf-8')))
    print(f'encoded: {encoded}')
    decoded = decode(encoded).tobytes().decode('utf-8')
    print(f'decoded: {decoded}')

    encoded.invert(3)  # introduce an error in a message bit
    encoded.pos = 0
    decoded = decode(encoded).tobytes().decode("utf-8")
    print(f'decoded after error in message bit: {decoded}')

    encoded.invert(12)  # introduce an error in a parity bit
    encoded.pos = 0
    decoded = decode(encoded).tobytes().decode("utf-8")
    print(f'decoded after error in parity check bit: {decoded}')
