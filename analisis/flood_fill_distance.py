import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt
from deques import Deque2D
from pathlib import Path

def rec_binflood_distance(binary_mask, x: int, y: int, xt: int, yt: int):     
    if x == xt and y == yt:
        return 0

    binary_mask[x,y] = False
    
    x_max, y_max = binary_mask.shape
    
    count = 1

    for dx, dy in ((+1, 0), (-1, 0), (0, -1), (0, +1)):
        nx, ny = x + dx, y + dy
        if not (0 <= nx < x_max):
            continue
        if not (0 <= ny < y_max):
            continue
        if not binary_mask[nx, ny]:
            continue
        count += binflood_distance(binary_mask, nx, ny, xt, yt)
    return count

@nb.njit(cache=True)
def breadth_binflood_distance(binary_mask, x: int, y: int, xt: int, yt: int, maxqlen:int = 4_000_000):     
    distance_mask = np.zeros(binary_mask.shape)
    
    if x == xt and y == yt:
        return 0, distance_mask
    binary_mask[x,y] = False
    queued_mask = np.full(binary_mask.shape, False)
    queue = Deque2D(maxqlen)
    queue.append(np.array([x,y]))

    x_max, y_max = binary_mask.shape
    
    for n in range(maxqlen/4):
        x, y = queue.popleft()

        for dx, dy in ((+1, 0), (-1, 0), (0, -1), (0, +1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < x_max):
                continue
            if not (0 <= ny < y_max):
                continue
            if not binary_mask[nx, ny]:
                continue
            if queued_mask[nx, ny]:
                continue
            binary_mask[x,y] = False
            #count = max(count, queue.count)
            #print(count)
            #print(f'(nx,ny)=({nx},{ny}): cnt = {count}')
            queue.append(np.array([nx,ny]))
            queued_mask[nx,ny] = True
            distance_mask[nx,ny] = distance_mask[x,y] + 1
            #print(f'queue = {queue}')
            if nx == xt and ny == yt:
                return distance_mask[nx,ny], distance_mask
    return (None, None) # Breadth first deadlock

@nb.njit(cache=True)
def second_nearest_neighbors(binary_mask, x: int, y: int, xt: int, yt: int, maxqlen:int = 4_000_000):     
    distance_mask = np.zeros(binary_mask.shape)
    
    if x == xt and y == yt:
        return 0, distance_mask
    binary_mask[x,y] = False
    queued_mask = np.full(binary_mask.shape, False)
    queue = Deque2D(maxqlen)
    queue.append(np.array([x,y]))

    x_max, y_max = binary_mask.shape
    
    for n in range(maxqlen/4):
        x, y = queue.popleft()

        for dx, dy in ((+1, 0), (-1, 0), (0, -1), (0, +1), (-1,-1), (-1,1), (1,-1), (1,1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < x_max):
                continue
            if not (0 <= ny < y_max):
                continue
            if not binary_mask[nx, ny]:
                continue
            if queued_mask[nx, ny]:
                continue
            binary_mask[x,y] = False
            #count = max(count, queue.count)
            #print(count)
            #print(f'(nx,ny)=({nx},{ny}): cnt = {count}')
            queue.append(np.array([nx,ny]))
            queued_mask[nx,ny] = True
            distance_mask[nx,ny] = distance_mask[x,y] + 1
            #print(f'queue = {queue}')
            if nx == xt and ny == yt:
                return distance_mask[nx,ny], distance_mask
    return (None, None) # Breadth first deadlock


@nb.njit(cache=True)
def flood_convolve_step(matrix_in, binary_mask):
    matrix_out = np.zeros_like(matrix_in)
    x_max, y_max = matrix_in.shape

    initval = x_max * y_max + 1 # Max possible distance/area
    
    for nx in range(1, x_max - 1):
        for ny in range(1, y_max - 1):
            if matrix_in[nx,ny] > 0:
                continue
            val = initval
            for dx, dy in ((+1,0), (0,+1), (-1,0),(0,-1)):
                if not (0 <= nx + dx < x_max):
                    continue
                if not (0 <= ny + dy < y_max):
                    continue
                if not binary_mask[nx+dx, ny+dy]:
                    continue
                val = np.minimum(matrix_in[nx+dx, ny+dy], val)

                if val > 0:
                    matrix_out[nx,ny] = val + 1
    return matrix_in + matrix_out

@nb.njit(cache=True)
def flood_convolve(binary_mask, x: int, y: int, xt: int, yt: int, maxlen: int=10_000_000):
    distance_matrix = np.zeros(binary_mask.shape, dtype=np.int64)
    distance_matrix[x,y] = 1
    
    for n in range(maxlen):
        print(n)
        distance_matrix = flood_convolve_step(distance_matrix, binary_mask)

        if distance_matrix[xt, yt] != 0:
            return distance_matrix[xt, yt] - 1
    return -1

@nb.njit(cache=True)
def random_points_normalized_flood_distance(binary_mask):
    """Deprecated!"""
    list_true = list(zip(np.where(binary_mask)[0], np.where(binary_mask)[1]))
    
    ran = np.random.choice(len(list_true))
    x, y = list_true[ran]
    ran = np.random.choice(len(list_true))
    xt, yt = list_true[ran]
    
    cp = np.copy(binary_mask)
    
    flood_dist, _ = breadth_binflood_distance(cp, x, y, xt, yt)
    rel_point = np.array([float(x-xt), float(y-yt)])
    dist = np.linalg.norm(rel_point)
    norm_dist = flood_dist / dist

    print(f'xt, yt = {xt}, {yt}')
    print(f'x, y = {x}, {y}')
    print(norm_dist)
    return norm_dist

def flood_distance_monte_carlo(binary_mask, N:int = 1000):
    out = np.zeros(N)
    for n in range(N):
        out[n] = random_points_normalized_flood_distance(binary_mask)
    return out

@nb.njit(cache=True)
def random_points_distance(binary_mask):
    list_true = list(zip(np.where(binary_mask)[0], np.where(binary_mask)[1]))
    
    ran = np.random.choice(len(list_true))
    x, y = list_true[ran]
    ran = np.random.choice(len(list_true))
    xt, yt = list_true[ran]
    
    cp = np.copy(binary_mask)
    
    flood_dist, _ = breadth_binflood_distance(cp, x, y, xt, yt)
    rel_point = np.array([float(x-xt), float(y-yt)])
    euclid_dist = np.linalg.norm(rel_point)
    manhattan_dist = np.abs(rel_point[0]) + np.abs(rel_point[1])

    return x, y, xt, yt, flood_dist, euclid_dist, manhattan_dist

def monte_carlo_distance(binary_mask, N:int = 1000):
    x = np.zeros(N, dtype=int)
    y = np.zeros(N, dtype=int)
    xt = np.zeros(N, dtype=int)
    yt = np.zeros(N, dtype=int)

    flood_dist = np.zeros(N)
    euclid_dist = np.zeros(N)
    manhattan_dist = np.zeros(N)

    for n in range(N):
        x[n], y[n], xt[n], yt[n], flood_dist[n], euclid_dist[n], manhattan_dist[n] = random_points_distance(binary_mask)
        # TODO: black formatter!

    df = pd.DataFrame(
    {
        "x" : x,
        "y" : y,
        "xt" : xt,
        "yt" : yt,
        "D_flood" : flood_dist,
        "D_manhattan" : manhattan_dist,
        "D_euclidean" : euclid_dist
    }
    )
    return df
