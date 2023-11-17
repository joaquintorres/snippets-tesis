import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import argparse
from pathlib import Path

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


def gen_fields(path: Path):
    sums = []
    means = []
    medians = []
    stds = []
    iqrs = []
    areas = []
    
    pores = []
    
    for pickle in path.glob('*.pickle'):
        df = pd.read_pickle(pickle)

        pores.append(pickle.name.removesuffix(".pickle"))
        sums.append(df["sum"][0])
        means.append(df["mean"][0])
        medians.append(df["median"][0])
        stds.append(df["std"][0])
        iqrs.append(df["iqr"][0])
        areas.append(df["area"][0])

    sums = np.array(sums)
    means = np.array(means)
    medians = np.array(medians)
    stds = np.array(stds)
    iqrs = np.array(iqrs)
    areas = np.array(areas)

    return pores, sums, means, medians, stds, iqrs, areas

def categorize_sums(pores, sums, areas):
    max_number = np.max(sums/areas)
    step_size = max_number / 5
    #bins = np.arange(0, max_number + step_size, step_size)
    bins = [0, step_size, step_size*2, step_size * 3, step_size * 4, max_number + 1] 
     
    categories = np.digitize(sums/areas, bins) 

    image_categories = {i: [] for i in range(1,6)}

    # Populate the dictionary with image names
    for i, category in enumerate(categories, start=1):
        image_categories[category].append(pores[i - 1])

    # Print the generated dictionary
    #for category, images in image_categories.items():
    #    print(f"Category {category}: {images}")
    
    return image_categories

def categorize_means_stds(pores, means, stds, areas, normalized:bool = False):
    if normalized:
        means_in = np.copy(means) / areas
        stds_in = np.copy(stds) / areas
    else:
        means_in = np.copy(means)
        stds_in = np.copy(stds)
    
    max_means = np.max(means_in)
    step_means = max_means / 5

    max_stds = np.max(stds_in)
    step_stds = max_stds / 5

    bins_means = [0, step_means, step_means * 2, step_means * 3, step_means * 4, max_means + 1] 
    bins_stds = [0, step_stds, step_stds * 2, step_stds * 3, step_stds * 4, max_stds + 1] 

    image_categories = {(i,j): [] for i in range(1,6) for j in range(1,6)}

    categories_means = np.digitize(means_in, bins_means)
    categories_stds = np.digitize(stds_in, bins_stds)

    for i, category_mean in enumerate(categories_means, start=1):
        for j, category_std in enumerate(categories_stds, start=1):
            if pores[i - 1] == pores[j -1]:
                image_categories[(category_mean, category_std)].append(pores[i - 1])

    #for category, images in image_categories.items():
    #    print(f"Category {category}: {images}")
    
    return image_categories

def categorize_medians_iqrs(pores, medians, iqrs, areas, normalized:bool = False):
    if normalized:
        medians_in = np.copy(medians) / areas
        iqrs_in = np.copy(iqrs) / areas
    else:
        medians_in = np.copy(medians)
        iqrs_in = np.copy(iqrs)
    
    max_medians = np.max(medians_in)
    step_medians = max_medians / 5

    max_iqrs = np.max(iqrs_in)
    step_iqrs = max_iqrs / 5

    bins_medians = [0, step_medians, step_medians * 2, step_medians * 3, step_medians * 4, max_medians + 1] 
    bins_iqrs = [0, step_iqrs, step_iqrs * 2, step_iqrs * 3, step_iqrs * 4, max_iqrs + 1] 

    image_categories = {(i,j): [] for i in range(1,6) for j in range(1,6)}

    categories_medians = np.digitize(medians_in, bins_medians)
    categories_iqrs = np.digitize(iqrs_in, bins_iqrs)

    for i, category_median in enumerate(categories_medians, start=1):
        for j, category_iqr in enumerate(categories_iqrs, start=1):
            if pores[i - 1] == pores[j -1]:
                image_categories[(category_median, category_iqr)].append(pores[i - 1])

    #for category, images in image_categories.items():
    #    print(f"Category {category}: {images}")
    
    return image_categories

def plot_sums(image_dict: dict):
    IMAGE_DIR = Path("/home/torres/git/tesis/peru2003-pores")
    
    fig, ax = plt.subplots(ncols=5)

    for m in range(1, 6):
        ims = image_dict[m]
        print(ims)
        filename = IMAGE_DIR/(ims[0].removesuffix("_filled") + ".npy")
        image = np.load(filename)
        idx = crop_pore(image)
    
        ax[m-1].imshow(image[idx])
        ax[m-1].set_title(f"Categoría {m}")
    #plt.savefig("sums-out", dpi=300)
    plt.show()
    plt.close()

def plot_means_stds(image_dict: dict, normalized: bool=False):
    IMAGE_DIR = Path("/home/torres/git/tesis/peru2003-pores")
    
    fig, ax = plt.subplots(ncols=5, nrows=5)
    plt.subplot_tool(targetfig=fig)
    for m in range(1, 6):
        for n in range(1, 6):
            ims = image_dict[(m,n)]
            if not ims:
                ax[m-1, n-1].imshow(np.zeros((300,300)))
                ax[m-1, n-1].axis('off')
                continue

            filename = IMAGE_DIR/(ims[0].removesuffix("_filled") + ".npy")
            image = np.load(filename)
            idx = crop_pore(image)
        
            ax[m-1, n-1].imshow(image[idx])
            ax[m-1, n-1].set_title(f"Categoría ({m},{n})")

    if normalized:
        fig.suptitle("Media-desvío estándar normalizada")
        #plt.savefig("means-stds-norm-out", dpi=300)
        plt.show()
        plt.close()
    else:
        fig.suptitle("Media-desvío estándar")
        # plt.savefig("means-stds-out", dpi=300)
        plt.show()
        plt.close()

def plot_medians_iqrs(image_dict: dict, normalized: bool=False):
    IMAGE_DIR = Path("/home/torres/git/tesis/peru2003-pores")
    
    fig, ax = plt.subplots(ncols=5, nrows=5)
    plt.subplot_tool(targetfig=fig)

    for m in range(1, 6):
        for n in range(1, 6):
            ims = image_dict[(m,n)]
            if not ims:
                ax[m-1, n-1].imshow(np.zeros((300,300)))
                ax[m-1, n-1].axis('off')
                continue

            filename = IMAGE_DIR/(ims[0].removesuffix("_filled") + ".npy")
            image = np.load(filename)
            idx = crop_pore(image)
        
            ax[m-1, n-1].imshow(image[idx])
            ax[m-1, n-1].set_title(f"Categoría ({m},{n})")

    if normalized:
        fig.suptitle("Mediana-IQR normalizada")
        # plt.savefig("medians-iqrs-norm-out", dpi=300)
        plt.show()
        plt.close()
    else:
        fig.suptitle("Mediana-IQR estándar")
        # plt.savefig("medians-iqrs-out", dpi=300)
        plt.show()
        plt.close()

def plot_histograms(means, medians, stds, iqrs):
    fig, ax = plt.subplots(ncols=4)
    ax[0].hist(means)
    ax[0].set_title("Medias")
    ax[1].hist(stds)
    ax[1].set_title("Desvíos estándar")
    ax[2].hist(medians)
    ax[2].set_title("Medianas")
    ax[3].hist(iqrs)
    ax[3].set_title("IQRs")
    # plt.savefig("histograms", dpi=300)
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Process a directory path.")
    parser.add_argument("directory", type=str, help="Input directory path")

    args = parser.parse_args()
    directory_path = Path(args.directory)

    if not directory_path.exists() or not directory_path.is_dir():
        print("Error: Invalid directory path.")
        return

    print("Input directory:", directory_path)
        
    pores, sums, means, medians, stds, iqrs, areas = gen_fields(directory_path)
    #print(pores, sums, means, medians, stds, iqrs, areas)    
    
    plot_histograms(means=means, medians=medians, stds=stds, iqrs=iqrs)
    
    sums_dict = categorize_sums(pores=pores, sums=sums, areas=areas)
    plot_sums(sums_dict)

    means_stds_dict = categorize_means_stds(pores=pores, means=means, stds=stds, areas=areas)
    plot_means_stds(means_stds_dict)

    means_stds_dict = categorize_means_stds(pores=pores, means=means, stds=stds, areas=areas, normalized=True)
    plot_means_stds(means_stds_dict, normalized=True)
    
    medians_iqrs_dict = categorize_medians_iqrs(pores=pores, medians=medians, iqrs=iqrs, areas=areas)
    plot_medians_iqrs(medians_iqrs_dict)
    
    medians_iqrs_dict = categorize_medians_iqrs(pores=pores, medians=medians, iqrs=iqrs, areas=areas, normalized=True)
    plot_medians_iqrs(medians_iqrs_dict, normalized=True)


if __name__ == "__main__":
    main()
