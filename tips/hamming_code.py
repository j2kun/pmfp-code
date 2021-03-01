from bitstring import BitStream
from bitstring import Bits


encoder = {
    0: 0,
    1: 105,
    2: 42,
    3: 67,
    4: 76,
    5: 37,
    6: 102,
    7: 15,
    8: 112,
    9: 25,
    10: 90,
    11: 51,
    12: 60,
    13: 85,
    14: 22,
    15: 127
}

syndrome_table = {
    0: 0,
    1: 1 << 3,
    2: 1 << 5,
    3: 1 << 1,
    4: 1 << 6,
    5: 1 << 2,
    6: 1 << 4,
    7: 1,
}


def encode(bits: BitStream) -> BitStream:
    '''Encode a message to be resilient to errors.

    Arguments:
      - bits: a bit stream to encode, of length divisible by 4

    Returns:
      A bit stream containing the encoded message
    '''
    bits.pos = 0
    output = BitStream()
    for i in range(len(bits) // 4):
        block = bits.read('uint:4')
        output.append(Bits(uint=encoder[block], length=7))
    return output


def parity(x: int) -> int:
    # python 3.10 will have bit_count,
    # can replace this with (block & d).bit_count() & 1
    return Bits(uint=x, length=7).count(1) & 1


def decode(bits: BitStream) -> BitStream:
    '''Decode a message while also correcting any errors found.

    If the number of errors is too large, the decoded message may be incorrect.

    Arguments:
      - bits: the encoded stream of bits, in blocks of 7 bits

    Returns:
      A BitStream representing the decoded message
    '''
    bits.pos = 0
    output = BitStream()
    for i in range(len(bits) // 7):
        block = bits.read('uint:7')
        syndrome = (
            4 * parity(block & 85)
            + 2 * parity(block & 51)
            + parity(block & 15)
        )
        block ^= syndrome_table[syndrome]
        decoded = (
            ((block & 16) >> 1)
            + (block & 4)
            + (block & 2)
            + (block & 1)
        )
        output.append(Bits(uint=decoded, length=4))

    return output


if __name__ == "__main__":
    import os
    import timeit
    print('Generating 1 KiB of random data')
    msg = os.urandom(1024)  # 1 KiB

    unencoded_input = BitStream(msg)
    print('Timing encode')
    print(timeit.timeit(lambda: encode(unencoded_input), number=10) / 10)

    encoded = encode(unencoded_input)
    print('Timing decode')
    print(timeit.timeit(lambda: decode(encoded), number=10) / 10)
