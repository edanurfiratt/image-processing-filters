import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, filters
from scipy.ndimage import uniform_filter, median_filter

# Örnek görüntü
image = data.astronaut()

# Grayscale
gray = color.rgb2gray(image)

# Filtreler
average = uniform_filter(gray, size=3)
gaussian = filters.gaussian(gray, sigma=1)
sobel = filters.sobel(gray)
laplace = filters.laplace(gray)
median = median_filter(gray, size=3)

# Görselleştirme
fig, ax = plt.subplots(2, 3, figsize=(12, 8))

ax[0,0].imshow(gray, cmap='gray')
ax[0,0].set_title("Original")

ax[0,1].imshow(average, cmap='gray')
ax[0,1].set_title("Average")

ax[0,2].imshow(gaussian, cmap='gray')
ax[0,2].set_title("Gaussian")

ax[1,0].imshow(sobel, cmap='gray')
ax[1,0].set_title("Sobel")

ax[1,1].imshow(laplace, cmap='gray')
ax[1,1].set_title("Laplacian")

ax[1,2].imshow(median, cmap='gray')
ax[1,2].set_title("Median")

for a in ax.ravel():
    a.axis("off")

plt.tight_layout()
plt.savefig("output.png")
plt.show()