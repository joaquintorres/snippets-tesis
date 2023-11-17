import numpy as np
import numba as nb
import flood_fill_distance as ff
from flood_fill_distance import *
from shapely.geometry import LineString

#Generates the patterns, with a global variable.
# Instead of List[numpy arrays] generate a Set [transformed numpy arrays ex. np.array to string]
# pattern.to_string -> {"000000111", ...} Then compare.
paint_patterns = [np.array([[0., 0., 0.],
                        [0., 0., 0.],
                        [1., 1., 1.]]),
                  np.array([[0., 0., 0.],
                        [1., 0., 0.],
                        [1., 1., 1.]]),
                  np.array([[0., 0., 0.],
                        [0., 0., 1.],
                        [1., 1., 1.]]),
                  np.array([[0., 0., 0.],
                        [1., 0., 0.],
                        [1., 1., 0.]]),
                  np.array([[0., 0., 0.],
                        [0., 0., 1.],
                        [0., 1., 1.]]),
                  np.array([[0., 0., 0.],
                        [1., 0., 1.],
                        [1., 1., 1.]]),
                  np.array([[0., 0., 0.],
                        [1., 0., 0.],
                        [0., 1., 0.]]),
                  np.array([[0., 0., 0.],
                        [0., 0., 1.],
                        [0., 1., 0.]]),
                  np.array([[1., 1., 1.],
                        [1.,0.,1.],
                        [1., 1., 1.]]),
                 ]

paint_set = set()

for paint_pattern in paint_patterns:
    # all rotations
    for degree in range(4): 
        paint_set.add(np.rot90(paint_pattern, degree).tobytes())

def shortest_flood_path(distance_matrix, xt: int, yt: int, maxlen = 1_000_000):
    path = [(xt,yt)]
    x, y = xt, yt

    for _ in range(maxlen):
        for dx, dy in ((+1,0), (-1,0), (0,+1), (0,-1)):
            nx, ny = x + dx, y + dy

            if distance_matrix[nx,ny] == distance_matrix[x,y] - 1.0:
                path.append((nx,ny))
                x, y = nx, ny
                break

    return path

def perpendicular_vector(a: tuple[int, int], b: tuple[int, int], length: int=1):
    ab = LineString([a, b])
    left = ab.parallel_offset(length / 2, 'left')
    right = ab.parallel_offset(length / 2, 'right')
    c = left.boundary.centroid
    d = right.boundary.centroid  # note the different orientation for right offset
    return ((c.x, c.y), (d.x, d.y))

#@nb.njit(cache = True)
def perpendicular_line_length(binary_mask, a: tuple[int, int], b: tuple[int, int], maxlen:int = 50_000):
    vec = b[0]-a[0], b[1]-a[1]
    xp_old, yp_old = int(a[0]), int(a[1])
    dist = 0
        
    size_x, size_y = binary_mask.shape

    # Forwards
    for k in range(2, maxlen):
        xp, yp = int(k*vec[0] + a[0]), int(k*vec[1] + a[1])

        if not (0 <= xp < size_x):
            dist += np.linalg.norm(np.array([xp-a[0], yp-a[1]]))
            break
        if not (0 <= yp < size_y):
            dist += np.linalg.norm(np.array([xp-a[0], yp-a[1]]))
            break
        #print(f'xp {xp} yp {yp} k {k}')
        if binary_mask[xp,yp]:
        #   print("write!")
            binary_mask[xp,yp] = False
        else:
            dist += np.linalg.norm(np.array([xp-a[0], yp-a[1]]))
            break


    # Backwards
    for k in range(2, maxlen):
        xp, yp = int(-k*vec[0] + a[0]),int(-k*vec[1] + a[1])
        if not (0 <= xp < size_x):
            dist += np.linalg.norm(np.array([xp-a[0], yp-a[1]]))
            break
        if not (0 <= yp < size_y):
            dist += np.linalg.norm(np.array([xp-a[0], yp-a[1]]))
            break
        #print(f'xp {xp} yp {yp} k {k}')
        if binary_mask[xp,yp]:
        #   print("write!")
            binary_mask[xp,yp] = False
        else:
            dist += np.linalg.norm(np.array([xp-a[0], yp-a[1]]))
            break           
    
    return dist

def random_line_strangle(binary_mask, sample_size: int = 100, offset: int = 3):
    """Given a binary mask, selects 2 random points and samples <sample_size> 
    distances perpendicular to a random infinitesimal segment."""
    
    out = np.zeros(sample_size)
    list_true = list(zip(np.where(binary_mask)[0], np.where(binary_mask)[1]))

    ran = np.random.choice(len(list_true))
    x, y = list_true[ran]
    ran = np.random.choice(len(list_true))
    xt, yt = list_true[ran]

    cp = np.copy(binary_mask)

    ff_distance, distance_matrix = ff.breadth_binflood_distance(cp, x, y, xt, yt)
    short_path = shortest_flood_path(distance_matrix, xt, yt)
    for x, y in short_path:
        binary_mask[x,y] = False

    for n in range(sample_size):
        ran = np.random.choice(len(short_path) - offset) # Avoid bounds error w offset
        A = short_path[ran]
        B = short_path[ran+offset]
        a, b = perpendicular_vector(a=A, b=B)
        out[n] = perpendicular_line_length(np.copy(binary_mask), a=a, b=b)
    return out

def random_points_strangle(binary_mask, npoints: int = 100, offset: int = 3, strangle_metric: str = 'std'):
    """Given a binary mask, selects 2 random points and samples <sample_size> 
    distances perpendicular to a random infinitesimal segment.
    
    binary_mask: numpy array containing the corresponding binary mask.
    npoints: number of points to sample random paths and strangle values.
    strangle_metric: how the distribution is transformed into a strangle value.
        'std': Use standard deviation
        'mean': Use mean
        'median': Use median
    """
    
    sample_size = 10
    out = np.zeros(npoints)
    list_true = list(zip(np.where(binary_mask)[0], np.where(binary_mask)[1]))

    for m in range(npoints):
        # Pick two random points

        ran = np.random.choice(len(list_true))
        x, y = list_true[ran]
        ran = np.random.choice(len(list_true))
        xt, yt = list_true[ran]

        ff_distance, distance_matrix = ff.breadth_binflood_distance(np.copy(binary_mask), x, y, xt, yt)
        short_path = shortest_flood_path(distance_matrix, xt, yt)

        cp = np.copy(binary_mask)

        for x, y in short_path:
            cp[x,y] = False

        strangle_distr = np.zeros(sample_size)

        for n in range(sample_size):
            ran = np.random.choice(len(short_path) - offset) # Avoid bounds error w offset
            A = short_path[ran]
            B = short_path[ran+offset]
            a, b = perpendicular_vector(a=A, b=B)
            strangle_distr[n] = perpendicular_line_length(np.copy(cp), a=a, b=b)
    
        match strangle_metric:
            case 'std':
                out[m] = np.std(strangle_distr)
            case 'mean':
                out[m] = np.mean(strangle_distr)
            case 'median':
                out[m] = np.median(strangle_distr)
            case 'min':
                out[m] = np.min(strangle_distr)
            case _:
                raise ValueError("Not an acceptable metric")

    return out

def dilation_strangle(binary_mask, init_point: tuple[int, int], end_point: tuple[int, int], maxlen = 100):
    """Calculates strangulation by dilation of the shortest path.   """
    x, y = init_point
    xt, yt = end_point
    strangulation = 0

    for _ in range(maxlen):
    # Short Path
        try:
            dist, distance_matrix = ff.breadth_binflood_distance(np.copy(binary_mask), x, y, xt, yt)
        except ValueError:
            return strangulation

        short_path = shortest_flood_path(distance_matrix, xt, yt)

        distance_matrix = np.zeros(binary_mask.shape)
        level_matrix = binmask2array(binary_mask)

        for x, y in short_path:
            # binary_mask[x,y] = False
            level_matrix[x,y] = 1

        touchpoints = dilation(level_matrix, binary_mask)
        if touchpoints == None:
            raise ValueError("Faulty binary mask.")
       
        refpoint = get_refpoint(binary_mask, touchpoints)
    #    try:
        #    path_to_delete = touchpoint_minpath(binary_mask, short_path, refpoint)
        #except ValueError:
        #    return strangulation
        
        path_to_delete = touchpoint_minpath(binary_mask, short_path, refpoint)
        
        for x,y in path_to_delete:
            binary_mask[x,y] = False # level_matrix is a local variable
        strangulation += 1
        print(strangulation)

    return strangulation

def binmask2array(binary_mask):
    """Transform binary mask into a matrix with 0: False, 1: Path, 2: True"""
    out = np.zeros(binary_mask.shape)
    # out[~binary_mask] = 2
    return out

def get_kernel(matrix, point: tuple[int,int]):
    x, y = point

    return matrix[x-1:x+2,y-1:y+2]

def dilation_step(level_matrix, point: tuple[int, int]):
    kernel = get_kernel(level_matrix, point=point)
    # If we're in a corner of the screen, do nothing 
    if kernel.size < 9:
        return None

    byte_kernel = kernel.tobytes()

    # if 1 in kernel:
    #     print(kernel)
    #     print(byte_kernel in paint_set)
    if byte_kernel in paint_set:
        return point 

    return None # Careful there

def dilation(level_matrix, binary_mask, maxlen: int = 1000):
    cp = np.copy(level_matrix)
    for n in range(maxlen):
        xz, yz = np.where(level_matrix == 0)
        for x, y in zip(xz, yz):
            point = dilation_step(level_matrix, point=(x,y))
            if point != None:
                cp[point] = 1
        level_matrix = np.copy(cp)
        if n % 5 == 0:
            fig, ax = plt.subplots()
            ax.imshow(level_matrix.astype(bool) | (~binary_mask))
            fig.savefig(f'and-{n:03d}')
        if (~binary_mask & level_matrix.astype(bool)).any():
            xe, ye = np.where(binary_mask == level_matrix)
            tmp = np.zeros(binary_mask)
            tmp[binary_mask] = 1.
            tmp[xe, ye] = 2.
            fig, ax = plt.subplots()
            ax.imshow(tmp)
            fig.savefig('intersection')
            return list(zip(xe, ye))
    return None

def touchpoint_minpath(binary_mask, short_path: list[tuple[int, int]], refpoint: tuple[int, int]):
    points = np.array(short_path)
    reference = np.array(refpoint)
    x, y = reference

    distances = np.sum((points - reference)**2, axis=1) # Why sqrt if minimizing
    
    minpoint = points[np.argmin(distances)]
    
    xt, yt = minpoint
    _, distance_matrix = ff.breadth_binflood_distance(np.copy(binary_mask), x , y, xt, yt)
    minpath = shortest_flood_path(distance_matrix, xt, yt)

    return minpath

def get_refpoint(binary_mask, touchpoints: list[tuple[int, int]]):
    for touchpoint in touchpoints:
        x, y = touchpoint
        x_max, y_max = binary_mask.shape
        for  dx, dy in ((+1,0), (-1,0), (0,+1), (0,-1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < x_max):
                continue
            if not (0 <= ny < y_max):
                continue
            if binary_mask[nx,ny]:
                return (nx,ny)
    return (-1, -1)
