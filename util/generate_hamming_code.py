import numpy

# fmt: off
P = numpy.array([
    1, 1, 0,
    1, 0, 1,
    0, 1, 1,
    1, 1, 1,
], dtype=numpy.uint8).reshape((4, 3))
# fmt: on

generator_matrix = numpy.concatenate(
    (numpy.identity(n=4, dtype=numpy.uint8), P), axis=1
)

parity_check_matrix = numpy.concatenate(
    (P.transpose(), numpy.identity(n=3, dtype=numpy.uint8)), axis=1
)


if __name__ == "__main__":

    def to_vec(x, length):
        vector = numpy.zeros((length,), dtype=numpy.uint8)
        for bit in range(length):
            vector[length - bit - 1] = (x & (1 << bit)) >> bit
        return vector

    def from_vec(x):
        n = len(x)
        output = 0
        for i in range(n):
            output |= x[n - i - 1] << i
        return output

    print(generator_matrix)
    print(parity_check_matrix)
    print(parity_check_matrix.dot(generator_matrix.T) % 2)

    encoder_lookup = []
    for i in range(16):
        input_vector = to_vec(i, 4)
        encoded = input_vector.dot(generator_matrix) % 2
        print(f"i={i}={input_vector}, o={from_vec(encoded)}={encoded}")
        encoder_lookup.append(from_vec(encoded))
    print(f"encoder = {encoder_lookup}")

    decoder_bits = [0] * 3
    for i, row in enumerate(parity_check_matrix):
        row_as_int = from_vec(row)
        decoder_bits[i] = row_as_int

    print(
        "syndrome = ("
        f"4 * parity(block & {decoder_bits[0]}) "
        f"+ 2 * parity(block & {decoder_bits[1]}) "
        f"+ parity(block & {decoder_bits[2]}))"
    )

    syndrome_lookup = [0] * 8
    for i in range(7):
        error_bit = 1 << (6 - i)
        error_vector = numpy.zeros((7,), dtype=numpy.uint8)
        error_vector[i] = 1
        syndrome = parity_check_matrix.dot(error_vector) % 2
        lookup_input = from_vec(syndrome)
        syndrome_lookup[lookup_input] = error_bit
    print(f"syndrome_table = {syndrome_lookup}")
