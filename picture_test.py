# Author: Wren Taylor

import imageio as iio
import numpy as np
from matplotlib import pyplot
from skimage import color

# Load the image
image = iio.imread('imageio:chelsea.png')

# Convert the image to grayscale
img_gray = color.rgb2gray(image)

# Normalize the grayscale image
img_gray_normalized = img_gray / np.max(img_gray)


original_rank = np.linalg.matrix_rank(img_gray_normalized)

print("Rank of the original image:", original_rank)


# Perform QR decomposition
Q, R = np.linalg.qr(img_gray_normalized)

# Choose a rank to approximate the image (you can adjust this as needed)
rank = 50

# Truncate Q and R to the chosen rank
Q_truncated = Q[:, :rank]
R_truncated = R[:rank, :]

# Reconstruct the image using the truncated matrices
img_reconstructed = Q_truncated @ R_truncated

# Display the original grayscale image
pyplot.subplot(1, 2, 1)
pyplot.imshow(img_gray_normalized, cmap=pyplot.cm.gray)
pyplot.title('Original Grayscale Image')

# Display the reconstructed image
pyplot.subplot(1, 2, 2)
pyplot.imshow(img_reconstructed, cmap=pyplot.cm.gray)
pyplot.title('Reconstructed Image (Rank {})'.format(rank))

pyplot.show()

