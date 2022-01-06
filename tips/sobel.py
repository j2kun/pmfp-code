

# use numpy?
SOBEL_KERNEL = []


def sobel(image):
    return convolve(image, SOBEL_KERNEL)

def convolve(matrix, kernel):
    pass
