import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from pathlib import Path

in_path = Path("peru2003-pores")
out_path = Path("watershed-segmentation")

def crop_pore(image, frame_size=1):
    true_indices = np.where(image==True)

    if len(true_indices) == 0:
        return image

    # Calculate the bounding box of the True region
    min_row = np.min(true_indices[0])
    min_col = np.min(true_indices[1])
    max_row = np.max(true_indices[0])
    max_col = np.max(true_indices[1])

    # Calculate the new bounding box with the desired frame size
    #min_row = max(0, min_row - frame_size)
    #min_col = max(0, min_col - frame_size)
    #max_row = min(image.shape[0], max_row + frame_size + 1)
    #max_col = min(image.shape[1], max_col + frame_size + 1)

    return (slice(min_row, max_row), slice(min_col, max_col))


def filled_watershed(pore):
    cp = np.copy(pore)
    filled = binary_fill_holes(cp)
    cp_filled = np.copy(filled)
    
    distances = distance_transform_edt(cp_filled)
    coords = peak_local_max(distances, footprint=np.ones((3, 3)), labels=filled)
    mask = np.zeros(distances.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distances, markers, mask=cp_filled)

    return distances, labels, filled

def pure_watershed(pore):
    cp = np.copy(pore)
    
    distances_unfilled = distance_transform_edt(cp)
    coords_unfilled = peak_local_max(distances_unfilled, footprint=np.ones((3, 3)), labels=cp)
    mask_unfilled = np.zeros(distances_unfilled.shape, dtype=bool)
    mask_unfilled[tuple(coords_unfilled.T)] = True
    markers_unfilled, _ = ndi.label(mask_unfilled)
    labels_unfilled = watershed(-distances_unfilled, markers_unfilled, mask=cp)
    
    return distances_unfilled, labels_unfilled

def plot_pore(pore, distances, labels, out:Path, variant: str="unfilled"):
    if variant == "filled":
        pore_label = "Poro relleno"
    else:
        pore_label = "Poro sin rellenar"
    
    idx = crop_pore(pore)
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(pore[idx], cmap=plt.cm.gray)
    ax[0].set_title(pore_label)
    ax[1].imshow(distances[idx], cmap=plt.cm.gray)
    ax[1].set_title('Transformada de Distancia Euclídea')
    ax[1].set_xlabel(f'suma={np.sum(distances)/np.sum(pore)}')
    ax[2].imshow(labels[idx], cmap=plt.cm.nipy_spectral)
    ax[2].set_title('Segmentación watershed')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.savefig(out)
    plt.close()

def gen_statistics(pore, distances):
    from scipy.stats import iqr
    
    idx = np.nonzero(distances)

    sum = np.sum(distances[idx])
    mean = np.mean(distances[idx])
    median = np.median(distances[idx])
    
    std = np.std(distances[idx])
    interquartile = iqr(distances[idx].flatten(), axis=0) 

    area = np.sum(pore)
    results = {"sum":sum, "mean": mean, "median": median, "std": std, "iqr": interquartile, "area": area}
    return results
    
SIZE_THRESHOLD = 15_000


for path in in_path.glob("*.npy"):
    pore = np.load(path)
    if np.sum(pore) < SIZE_THRESHOLD:
        continue
    
    distances, labels, filled = filled_watershed(pore)
    distances_unfilled, labels_unfilled = pure_watershed(pore)

    plot_pore(filled, distances, labels, out_path/(path.name.removesuffix(".npy")+"_filled"), variant="filled")
    plot_pore(pore, distances_unfilled, labels_unfilled, out_path/path.name.removesuffix(".npy"))
    
    results_filled = gen_statistics(pore, distances)
    results_unfilled = gen_statistics(pore, distances_unfilled)
    
    df_unfilled = pd.DataFrame(results_unfilled, index=[0])
    df_filled = pd.DataFrame(results_filled, index=[0])

    df_filled.to_pickle(out_path/(path.name.removesuffix(".npy")+"_filled"+".pickle"))
    df_unfilled.to_pickle(out_path/(path.name.removesuffix(".npy")+".pickle"))
