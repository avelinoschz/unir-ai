import cv2 as cv
from matplotlib import pyplot as plt

def imshow(title = 'imshow image', image = None, size = 10):
    h, w = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize = (size * aspect_ratio, size))
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

img_dir = './notebooks/images/img1.jpg'
img = cv.imread(img_dir)

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('image')
plt.show()

# second example, using a function
# imshow(image=img)

