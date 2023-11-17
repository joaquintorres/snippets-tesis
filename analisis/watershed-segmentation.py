import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

pore = np.load("peru2003-pores/1086.npy")
cp = np.copy(pore)
filled = binary_fill_holes(cp)
cp_filled = np.copy(filled)

distances = distance_transform_edt(cp_filled)
coords = peak_local_max(distances, footprint=np.ones((3, 3)), labels=filled)
mask = np.zeros(distances.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distances, markers, mask=cp_filled)

distances_unfilled = distance_transform_edt(cp)
coords_unfilled = peak_local_max(distances_unfilled, footprint=np.ones((3, 3)), labels=cp)
mask_unfilled = np.zeros(distances_unfilled.shape, dtype=bool)
mask_unfilled[tuple(coords_unfilled.T)] = True
markers_unfilled, _ = ndi.label(mask)
labels_unfilled = watershed(-distances_unfilled, markers_unfilled, mask=cp)



fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(cp, cmap=plt.cm.gray)
ax[0].set_title('Poro sin rellenar')
ax[1].imshow(distances_unfilled, cmap=plt.cm.gray)
ax[1].set_title('Transformada de Distancia Euclídea')
ax[2].imshow(labels_unfilled, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Segmentación watershed')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(filled, cmap=plt.cm.gray)
ax[0].set_title('Poro rellenado')
ax[1].imshow(distances, cmap=plt.cm.gray)
ax[1].set_title('Transformada de Distancia Euclídea')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Segmentación watershed')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()

