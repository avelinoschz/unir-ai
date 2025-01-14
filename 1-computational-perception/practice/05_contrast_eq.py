import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def histogram_equalization(image):
    """
    Perform histogram equalization on a grayscale image.

    :param image: 2D NumPy array representing a grayscale image.
    :return: 2D NumPy array of the equalized image.
    """
    # Flatten the image to calculate the histogram
    flat_image = image.flatten()
    
    # Calculate the histogram
    histogram = np.zeros(256, dtype=int)
    for pixel in flat_image:
        histogram[pixel] += 1

    # Calculate the cumulative distribution function (CDF)
    cdf = np.cumsum(histogram)
    
    # Normalize the CDF to map pixel values between 0 and 255
    cdf_min = cdf[np.nonzero(cdf)].min()  # Minimum non-zero CDF value

    cdf_normalized = (cdf - cdf_min) / (flat_image.size - cdf_min) * 255
    cdf_normalized = cdf_normalized.astype(np.uint8)

    # Map the original pixel values to the equalized values
    equalized_image = cdf_normalized[image]

    return equalized_image

if __name__ == "__main__":
    img_path = './notebooks/images/img1_contrast.jpg'

    img_contrasted = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img_equalized = histogram_equalization(img_contrasted)

	# Parameters:
    # clipLimit: Threshold for limiting contrast amplification.
    # tileGridSize: Size of the grid for the adaptive equalization.
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_contrasted)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(img_contrasted, cv.COLOR_BGR2RGB))
    plt.title("Image 2: Contrasted")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv.cvtColor(img_equalized, cv.COLOR_BGR2RGB))
    plt.title("Image 2: Histogram Eq")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(img_clahe, cv.COLOR_BGR2RGB))
    plt.title("Image 2: CLAHE")
    plt.axis('off')

    plt.show()

    hist_contrasted = cv.calcHist([img_contrasted], [0], None, [256], [0, 256])
    hist_equalized = cv.calcHist([img_equalized], [0], None, [256], [0, 256])
    hist_clahe = cv.calcHist([img_clahe], [0], None, [256], [0, 256])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(hist_contrasted)
    plt.title("Histogram 1: High Contrast")
    plt.xlim([0, 256])

    plt.subplot(1, 3, 2)
    plt.plot(hist_equalized)
    plt.title("Histogram 2: Histogram Eq")
    plt.xlim([0, 256])

    plt.subplot(1, 3, 3)
    plt.plot(hist_clahe)
    plt.title("Histogram 2: CLAHE")
    plt.xlim([0, 256])

    plt.show()

# Technique	    Use Case	                    Advantages	                                        Disadvantages
# Histogram     Equalization	                Global contrast enhancement	Simple and effective	Can amplify noise and artifacts
# CLAHE	        Uneven lighting, noise control  Avoids noise amplification, enhances local details	Requires parameter tuning

# Images with uneven lighting: Use CLAHE for localized adjustments.
