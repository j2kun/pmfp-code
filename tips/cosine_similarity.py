"""An implementation of cosine similarity."""


def cosine_similarity(a, b):
    dot_product = 0
    a_norm = 0
    b_norm = 0
    for a_i, b_i in zip(a, b):
        dot_product += a_i * b_i
        a_norm += a_i * a_i
        b_norm += b_i * b_i
    return dot_product / (a_norm**0.5 * b_norm**0.5)
