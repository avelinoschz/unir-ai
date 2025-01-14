import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def show_two_images(img1, img_title, img2, img2_title):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.title(img_title)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.title(img2_title)
    plt.axis('off')

    plt.show()

def add_salt_pepper_noise(image, prob):
    white = 255
    black = 0

    noisy_image = image.copy()
    random_matrix = np.random.random(image.shape)

    noisy_image[random_matrix < prob/2] = black
    noisy_image[random_matrix > (1 - prob/2)] = white

    return noisy_image

# 1. Median Filter (cv2.medianBlur)
# 	•	Best for: Salt-and-pepper noise.
# 	•	How it works:
# 	•	Replaces each pixel with the median value of its neighborhood.
# 	•	Effectively removes small, isolated noise without blurring edges.

# 2. Gaussian Blur (cv2.GaussianBlur)
# 	•	Best for: Gaussian noise (random variations in intensity).
# 	•	How it works:
# 	•	Applies a Gaussian kernel to smooth the image.
# 	•	Reduces high-frequency noise while preserving edges better than simple averaging.

# 3. Bilateral Filter (cv2.bilateralFilter)
# 	•	Best for: Removing noise while preserving edges.
# 	•	How it works:
# 	•	A combination of domain filtering and range filtering.
# 	•	Smooths regions with similar pixel intensities but maintains sharp edges.

def median_filter_3x3(image):
    """
    Apply a 3x3 median filter to a grayscale image using NumPy.

    :param image: 2D numpy array representing the grayscale image.
    :return: 2D numpy array representing the filtered image.
    """
    # Get the dimensions of the image
    height, width = image.shape

    # Create a padded version of the image to handle borders
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)

    # Create an empty array to store the result
    # los ceros permiten un mejor manejo de los efectos secundarios del padding
    filtered_image = np.zeros_like(image)

    # Perform the median filtering
    # se empieza tanto en columnas como en filas, por el padding
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            # Extract the 3x3 region
            region = padded_image[i-1:i+2, j-1:j+2]
            # Compute the median of the region
            filtered_image[i-1, j-1] = np.median(region)

    return filtered_image

if __name__ == "__main__":
    img1_path = './notebooks/images/img1.jpg'
    img2_path = './notebooks/images/img2.jpg'

    img1_gs = cv.imread(img1_path,  cv.IMREAD_GRAYSCALE)
    img2_gs = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)

    show_two_images(img1_gs, "Image 1: Original", img2_gs, "Image 2: Original")

    salt_pepper_prob = 0.10
    img1_noisy = add_salt_pepper_noise(img1_gs, salt_pepper_prob)
    img2_noisy = add_salt_pepper_noise(img2_gs, salt_pepper_prob)

    show_two_images(img1_noisy, "Image 1: Salt & Pepper Noise", img2_noisy, "Image 2: Salt & Pepper Noise")

    # bilateral
    # 9: Diameter of the filter.
	# 75, 75: Sigma values for intensity and spatial smoothing.

    # img1_filtered_gauss = cv.bilateralFilter(img1_noisy, 9, 75, 75)
    # img2_filtered_gauss = cv.bilateralFilter(img2_noisy,9, 75, 75)

    img1_filtered_gauss = cv.GaussianBlur(img1_noisy, (5, 5), 0)
    img2_filtered_gauss = cv.GaussianBlur(img2_noisy, (5, 5), 0)

    show_two_images(img1_filtered_gauss, "Image 1: Gaussian Filter", img2_filtered_gauss, "Image 2: Gaussian Filter")

    psnr1_gauss = cv.PSNR(img1_noisy, img1_filtered_gauss)
    psnr2_gauss = cv.PSNR(img2_noisy, img2_filtered_gauss)

    print(f"Image 1 Gaussian filter PSNR: {psnr1_gauss}")
    print(f"Image 2 Guassian filter PSNR: {psnr2_gauss}")

    img1_filtered_median = median_filter_3x3(img1_noisy)
    img2_filtered_median = median_filter_3x3(img2_noisy)

    show_two_images(img1_filtered_median, "Image 1: Median Filter", img2_filtered_median, "Image 2: Median Filter")

    psnr1_median = cv.PSNR(img1_noisy, img1_filtered_median)
    psnr2_median = cv.PSNR(img2_noisy, img2_filtered_median)

    print(f"Image 1 Median filter PSNR: {psnr1_median}")
    print(f"Image 2 Median filter PSNR: {psnr1_median}")

    # Definition: Measures the ratio of the maximum possible power of a signal (image) to the power of noise, expressed in decibels (dB). Higher PSNR indicates better filtering.


    # filtered = median_filter_3x3(image)

    # reasons to choose
	# •	PSNR:
	# •	Built into OpenCV.
	# •	Requires minimal setup and computation.
	# •	Works best if you only care about pixel-level differences.

    # The output of PSNR (Peak Signal-to-Noise Ratio) is a single scalar value, typically measured in decibels (dB). It quantifies the similarity between two images, with higher values indicating closer similarity and better quality.

    # Key Points About PSNR Output:
	# 1.Units:
	# •	PSNR is expressed in decibels (dB), a logarithmic scale that measures the ratio between the maximum possible power of a signal (image) and the power of noise.
	# 2.Typical Ranges:
	# •	30–50 dB: Good quality, depending on the application.
	# •	>50 dB: Almost identical images.
	# •	<30 dB: Significant difference or degradation.
	# 3. Interpretation:
	# •	A higher PSNR value means the filtered image is closer to the original (ground truth).
	# •	A lower PSNR value indicates that the filtered image has more noise or distortions.
    
    # PSNRExample Interpretation:

    # Scenario 1: High PSNR (45 dB)
    #     •	Filtered image closely matches the original.
    #     •	Noise removal was highly effective with minimal distortion.

    # Scenario 2: Low PSNR (20 dB)
    #     •	Filtered image differs significantly from the original.
    #     •	Noise removal was poor, or the image was over-smoothed/distorted.


    # APA
    # Gonzalez, R. C., & Woods, R. E. (2018). Digital image processing (4th ed.). Pearson. ISBN: 978-0133356724


    # The PSNR values indicate how similar the filtered images are to their respective original images. Here’s an interpretation of the results:

    # Bilateral Filter PSNRs:
	# 1.	Image 1 (20.60 dB):
	# •	The bilateral filter achieved a moderate PSNR value. This suggests that the noise reduction was significant but may have caused some loss of details or slight distortion.
	# 2.	Image 2 (22.18 dB):
	# •	The bilateral filter for Image 2 performed slightly better than for Image 1, resulting in a higher PSNR value. This indicates that the filter preserved details better while reducing noise.

    # Median Filter PSNRs:
	# 3.	Image 1 (14.96 dB):
	# •	The custom median filter had a significantly lower PSNR compared to the bilateral filter for the same image. This implies that while the noise was reduced, the filter may have overly smoothed the image, resulting in loss of details or edge blurring.
	# 4.	Image 2 (14.35 dB):
	# •	The median filter for Image 2 also resulted in a low PSNR, similar to Image 1. This indicates that the filter introduced significant artifacts or smoothed out fine details excessively.

    # Comparison Between Filters:
	# 1.	Bilateral vs. Median Filter:
	# •	The bilateral filter outperformed the median filter for both images, as evidenced by the higher PSNR values. This suggests that the bilateral filter was more effective at balancing noise reduction and detail preservation.
	# •	The custom median filter might require optimization (e.g., better handling of edges or fine details).
	# 2.	Image-Specific Performance:
	# •	For both filters, Image 2 consistently achieved a higher PSNR than Image 1, indicating that Image 2 may have had less noise initially or that it was more amenable to filtering.

    # Conclusions:
	# •	Bilateral Filter: Generally better for reducing noise while preserving details, as shown by higher PSNR values.
	# •	Median Filter: Likely oversmoothed the images, leading to lower PSNR values and possible loss of detail.
	# •	Recommendation: Optimize the custom median filter for specific noise types or use the bilateral filter for better overall performance.