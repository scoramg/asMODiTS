import numpy as np
import numba
from numba import cuda, prange

@numba.jit(nopython=True)
def numba_dist(a, b):
    #dist = np.zeros(a.shape[0])
    #for r in range(a.shape[0]):
    #    for c in range(128):
    #        dist[r] += (b[c] - a[r, c])**2
    #return dist (b - a)**2
    return (b - a)**2

@numba.jit(nopython=True, parallel=True, nogil=True)
def get_dtw_distance(dataset1, dataset2, w): 
    """
    Computes the dataset DTW distance matrix using multiprocessing.

    Args:
        dataset1: timeseries dataset of shape [N1, T1]
        dataset2: timeseries dataset of shape [N2, T2]

    Returns:
        Distance matrix of shape [N1, N2]
    """
    n1 = dataset1.shape[0]
    n2 = dataset2.shape[0]
    dist = np.empty((n1, n2), dtype=np.float64)

    for i in prange(n1):
        for j in prange(n2):
            dist[i][j] = dtw_sakoe_chiba_cuda(dataset1[i], dataset2[j], w)

    return dist

@cuda.jit
def dtw_sakoe_chiba_kernel(X, Y, band_width, dtw_matrix):
    i, j = cuda.grid(2)

    if i >= dtw_matrix.shape[0] or j >= dtw_matrix.shape[1]:
        return

    if abs(i - j) <= band_width:
        cost = numba_dist(X[i], Y[j])  # Implement your distance function here
        dtw_matrix[i, j] = cost + min(
            dtw_matrix[i - 1, j],         # Insertion
            dtw_matrix[i, j - 1],         # Deletion
            dtw_matrix[i - 1, j - 1]      # Match
        )

@numba.jit(nopython=True)
def dtw_sakoe_chiba_cuda(X, Y, band_width):
    numba.cuda.select_device(0)
    size_x, size_y = len(X), len(Y)
    dtw_matrix = np.zeros((size_x, size_y), dtype=np.float32)

    threads_per_block = (16, 16)
    blocks_x = (size_x + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (size_y + threads_per_block[1] - 1) // threads_per_block[1]
    blocks = (blocks_x, blocks_y)

    dtw_sakoe_chiba_kernel[blocks, threads_per_block](X, Y, band_width, dtw_matrix)

    return dtw_matrix

# Example usage
if __name__ == '__main__':
    X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    Y = np.array([2, 3, 4, 5, 6, 7], dtype=np.float32)
    band_width = 1

    dtw_matrix = dtw_sakoe_chiba_cuda(X, Y, band_width)
    print(dtw_matrix)