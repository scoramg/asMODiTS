from __future__ import absolute_import, division, print_function, unicode_literals
import bisect
#import sys, os
#sys.path.append(os.path.abspath(os.path.join('.', 'src')))
from six.moves import xrange
from collections import defaultdict
from numba import jit
import numpy as np
#from dtw_gpu import GpuDistance

''' import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule '''

def fastdtw(x, y, radius=1, dist=lambda a, b: abs(a - b), numba=False, gpu=False):
    min_time_size = radius + 2

    if len(x) < min_time_size or len(y) < min_time_size:
        if gpu:
            pass
        elif numba:
            return dtw(x, y, dist=dist)
        else:     
            return dtw(x, y, dist=dist)

    x_shrinked = __reduce_by_half(x)
    y_shrinked = __reduce_by_half(y)
    distance, path = fastdtw(x_shrinked, y_shrinked, radius=radius, dist=dist)
    window = __expand_window(path, len(x), len(y), radius)
    if numba:
        return dtw_numba(x, y)
    else:
        return dtw(x, y, window, dist=dist)


def dtw(x, y, window=None, dist=lambda a, b: abs(a - b)):
    #print("window:", window)
    len_x, len_y = len(x), len(y)
    if window is None:
        window = [(i, j) for i in xrange(len_x) for j in xrange(len_y)]
    window = [(i + 1, j + 1) for i, j in window]
    D = defaultdict(lambda: [float('inf')])
    D[0, 0] = [0, 0, 0]
    for i, j in window:
        D[i, j] = min([D[i-1, j][0], i-1, j], [D[i, j-1][0], i, j-1], [D[i-1, j-1][0], i-1, j-1], key=lambda a: a[0])
        D[i, j][0] += dist(x[i-1], y[j-1])
    path = []
    i, j = len_x, len_y
    while not (i == j == 0):
        path.append((i-1, j-1))
        i, j = D[i, j][1], D[i, j][2]
    path.reverse()
    return (D[len_x, len_y][0], path)

''' def dtw_sc(x, y, band_width=0, dist=lambda a, b: abs(a - b)):
    n, m = len(x), len(y)
    dtw_matrix = np.zeros((n, m))

    # Initialize the first row and column
    dtw_matrix[0, 0] = dist(x[0],y[0])
    for i in range(1, n):
        dtw_matrix[i, 0] = dtw_matrix[i - 1, 0] + dist(x[i], y[0])
    for j in range(1, m):
        dtw_matrix[0, j] = dtw_matrix[0, j - 1] + dist(x[0], y[j])

    # Apply Sakoe-Chiba band constraint
    for i in range(1, n):
        for j in range(max(1, i - band_width), min(m, i + band_width)):
            distance = dist(x[i], y[j])
            dtw_matrix[i, j] = distance + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

    # Find the optimal warping path
    path = []
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_val = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
            if min_val == dtw_matrix[i - 1, j]:
                i -= 1
            elif min_val == dtw_matrix[i, j - 1]:
                j -= 1
            else:
                i -= 1
                j -= 1
    path.append((0, 0))
    path.reverse()

    return dtw_matrix[n - 1, m - 1], path '''

''' @jit(nopython=True, cache=True)
def dtw_numba_sakoe_chiba(x, y, band_width=0, dist=lambda a, b: abs(a - b)):
    """
    Compute DTW (Dynamic Time Warping) between two sequences with Sakoe-Chiba band constraint.

    Parameters:
        x (np.ndarray): The first input sequence.
        y (np.ndarray): The second input sequence.
        band_width (int): The width of the Sakoe-Chiba band constraint.

    Returns:
        Tuple[float, List[Tuple[int, int]]]: The DTW distance and the optimal warping path as a list of (i, j) tuples.
    """
    n, m = len(x), len(y)
    dp = np.zeros((n, m), dtype=np.float64)

    # Initialize the first row and column of the dynamic programming matrix
    dp[0, 0] = dist(x[0], y[0])
    for i in range(1, n):
        dp[i, 0] = dp[i - 1, 0] + dist(x[i], y[0])
    for j in range(1, m):
        dp[0, j] = dp[0, j - 1] + dist(x[0], y[j])

    # Fill in the rest of the matrix using the Sakoe-Chiba band constraint
    for i in range(1, n):
        for j in range(max(1, i - band_width), min(m, i + band_width + 1)):
            cost = dist(x[i], y[j])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    # Backtrack to find the optimal warping path
    path = [(n - 1, m - 1)]
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            prev = min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            if dp[i - 1, j - 1] == prev:
                i -= 1
                j -= 1
            elif dp[i - 1, j] == prev:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    
    path.reverse()

    return (dp[n - 1, m - 1], path) '''


def __reduce_by_half(x, process_type = "cpu"):
    if process_type == "gpu":
        ''' input_gpu = gpuarray.to_gpu(x)
        new_size = len(x) // 2
        output_gpu = gpuarray.empty(new_size, dtype=np.float32)
        kernel_code = """
            __global__ void reduce_to_half(const float *input, unsigned size, float *output) {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid < size) {
                    output[tid] = (input[tid * 2] + input[tid * 2 + 1]) / 2;
                }
            }
        """
        block_size = 128
        grid_size = (new_size + block_size - 1) // block_size

        mod = SourceModule(kernel_code)
        func = mod.get_function("reduce_to_half")

        func(input_gpu, np.uint32(new_size), output_gpu, block=(block_size, 1, 1), grid=(grid_size, 1))
        return output_gpu.get() '''
        pass
    else:
        return [(x[i//2] + x[1+i//2]) / 2 for i in xrange(0, len(x), 2)]
    


def __expand_window(path, len_x, len_y, radius):
    path_ = set(path)
    for i, j in path:
        for a, b in ((i + a, j + b) for a in xrange(-radius, radius+1) for b in xrange(-radius, radius+1)):
            path_.add((a, b))

    window_ = set()
    for i, j in path_:
        for a, b in ((i * 2, j * 2), (i * 2, j * 2 + 1), (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1)):
            window_.add((a, b))

    window = []
    start_j = 0
    for i in xrange(0, len_x):
        new_start_j = None
        for j in xrange(start_j, len_y):
            if (i, j) in window_:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j

    return window
