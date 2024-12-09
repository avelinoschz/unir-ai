import cv2 as cv

img_dir = './images/car.png'

img_original = cv.imread(img_dir)
img_grayscale = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)

height_orig, width_orig, channels_orig = img_original.shape
print('Original color image')
print(f'height: {height_orig}, width: {width_orig}, channels: {channels_orig}')

height_gs, width_gs = img_grayscale.shape
print('Grayscale image')
print(f'height: {height_gs}, width: {width_gs}')

cv.imshow('car in color', img_original)
cv.imshow('car in grayscale', img_grayscale)

cv.waitKey(0)
cv.destroyAllWindows()