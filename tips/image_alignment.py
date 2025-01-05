import numpy as np


def max_index2d(arr):
    return np.unravel_index(np.argmax(arr), arr.shape)


def align_images(image1, image2):
    """Aligns two greyscale images using the discrete Fourier transform.

    Args:
        image1: A greyscale image.
        image2: A second greyscale image.

    Returns:
        A tuple containing the (row, column) pixel offset required to transform
        image2 to align with image1. The values of row are in the range [-N/2, N/2),
        where N is the number of rows in the image. A shift that would go
        beyond the bounds of the image is considered a cyclic shift. The same is true
        for the column shift and the image width.
    """
    # Apply window function to reduce edge effects. From Wikipedia
    # (https://en.wikipedia.org/wiki/Window_function): "Two-dimensional windows
    # are commonly used in image processing to reduce unwanted high-frequencies
    # in the image Fourier transform."
    #
    # There appears to be little practical difference between the "np.hanning"
    # and "np.hamming" window methods. "The Hamming window is 92% Hann window
    # and 8% rectangular window." (https://dsp.stackexchange.com/a/56408/1658).
    #
    # R. Hovden, Y. Jiang, H. Xin, L.F. Kourkoutis (2015). "Periodic Artifact
    # Reduction in Fourier Transforms of Full Field Atomic Resolution Images".
    # Microscopy and Microanalysis. 21 (2): 436–441. arXiv:2210.09024
    window = np.hamming(image1.shape[0])[:, None] * np.hamming(image1.shape[1])[None, :]
    image1_windowed = image1 * window
    image2_windowed = image2 * window

    # Use double precision for FFT operations
    f1 = np.fft.fft2(image1_windowed).astype(np.complex128)
    f2 = np.fft.fft2(image2_windowed).astype(np.complex128)

    # Normalize cross-power spectrum with small epsilon to avoid division by zero
    eps = 1e-15
    cross_power = (f1 * f2.conj()) / (np.abs(f1 * f2.conj()) + eps)

    correlation_sum = np.real(np.fft.ifft2(cross_power))
    correlation_sum = np.fft.fftshift(correlation_sum)

    # Find the peak location and value
    max_loc = max_index2d(correlation_sum)
    peak_value = correlation_sum[max_loc]

    # Calculate the mean and std of the correlation surface
    mean_correlation = np.mean(correlation_sum)
    std_correlation = np.std(correlation_sum)

    # If the peak is not significantly higher than the background, assume no shift
    if (peak_value - mean_correlation) < 3 * std_correlation:
        return 0, 0

    shape = np.array(image1.shape)
    shift = np.array([max_loc[0], max_loc[1]]) - shape // 2

    return shift[0], shift[1]
