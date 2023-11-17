import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from pathlib import Path
from skimage.filters import threshold_minimum

@nb.njit
def random_walk(binary_mask, N=1000, max_len = 1_000_000_000):
    out = np.zeros(N)
    choices = np.array([-1,0,1])
    xt, yt = np.where(binary_mask)
    indices = list(zip(xt, yt))

    for nx in range(N):
        x, y = indices[np.random.choice(len(indices))]
        walk_len = 0
        while binary_mask[x,y] and walk_len < max_len:
            x += np.random.choice(choices)
            y += np.random.choice(choices)
            walk_len += 1
        out[nx]=walk_len
    return out

LETTERS_PATH = './letters'
for impath in Path(LETTERS_PATH).glob('*.png'):
    A = plt.imread(impath)
    #thresh_min = threshold_minimum(A[:,:,3])
    thresh_min = 0.5
    binary_mask = A[:,:,3] > thresh_min
    out = random_walk(binary_mask)
    np.save(impath.name.removesuffix(".png"), out)
