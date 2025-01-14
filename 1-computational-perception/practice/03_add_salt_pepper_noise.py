import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def imshow(title = 'imshow image', image = None, size = 10):
    h, w = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize = (size * aspect_ratio, size))
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

img_dir = './notebooks/images/img1.jpg'

salt_pepper_prob = 0.10

white = 255
black = 0

img_original = cv.imread(img_dir)
img_gs = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)

noisy_image = img_gs.copy()
    
random_matrix = np.random.random(img_gs.shape)

noisy_image[random_matrix < salt_pepper_prob/2] = black

noisy_image[random_matrix > (1 - salt_pepper_prob/2)] = white

imshow("noisy image", noisy_image)

# def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):

#sesion 4

# al kernel tambien se le llama filtro
# el kernel es una mascara con valores, que se aplica sobre la matriz de pixeles
# ya existen kernel con valores especificos dependiendo la tarea o el resultado que se busque

# kernel de sobel - deteccion de bordes. Todo se ve negro con los bordes en blanco
# hay uno para detectar principalmente lineas horizontales

# kernel de sharpening - para acentuar mas los bordes.

#los filtros no lineas se llaman asi porque no se trabaja elementoa a elemento a diferencia de usar kernel, que es 1 por 1.
#  Por ej el de la "mediana" es un filtro no lineal

# El filtro o kernel de promedio, genera un suavizamiento.
# Si el kernel es de 3x3, seria un kernel con 9 valores de 1/9

# TODO verificar si MSE vs PSNR
# hacer ruido con opencv?
# en lugar de hacer una copia de la imagen, hacer una con zeros
# guardar html luego a pdf? intentar pdf




# sesion 5
# operaciones basicas de imagenes. Suma, resta, multiplicacion
# resta de imagnes puede servir para deteccion de movimiento

# mejora de contraste es una reasignacion de niveles de intensidad
# tecnicas globales y tecnicas locales
# tecnica global va a tomar informacion de toda la imagen para dedistribuir los niveles de intensidad en un nueva imagen

# ecualizacion
# se sacan las freq de niveles de intensidad (histograma)

# PDF, probabilidad de distribucion de freq (dividir el num de freq / total de freq)
# CDF, distribucion de freq acumulada (ir sumando cada PDF, hasta llegar a 1)

# TODO ver el proceso de eq de imagen del video sesion 5

