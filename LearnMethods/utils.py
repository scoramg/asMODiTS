import numpy as np

from Utils.timeseries import TimeSeriesResampler
from scipy.interpolate import interp1d

class EmptyClusterError(Exception):
    def __init__(self, message=""):
        super().__init__()
        self.message = message

    def __str__(self):
        if len(self.message) > 0:
            suffix = " (%s)" % self.message
        else:
            suffix = ""
        return "Cluster assignments lead to at least one empty cluster" + \
               suffix

def _check_no_empty_cluster(labels, n_clusters):
    """Check that all clusters have at least one sample assigned.
    """

    for k in range(n_clusters):
        if np.sum(labels == k) == 0:
            raise EmptyClusterError
        
def _check_full_length(centroids):
    """Check that provided centroids are full-length (ie. not padded with
    nans).

    If some centroids are found to be padded with nans, TimeSeriesResampler is
    used to resample the centroids.
    """
    resampler = TimeSeriesResampler(sz=centroids.shape[1])
    return resampler.fit_transform(centroids)

def _compute_inertia(distances, assignments, squared=True):
    """Derive inertia (average of squared distances) from pre-computed
    distances and assignments.

    Examples
    --------
    >>> dists = numpy.array([[1., 2., 0.5], [0., 3., 1.]])
    >>> assign = numpy.array([2, 0])
    >>> _compute_inertia(dists, assign)
    0.125
    """
    n_ts = distances.shape[0]
    if squared:
        return np.sum(distances[np.arange(n_ts),
                                   assignments] ** 2) / n_ts
    else:
        return np.sum(distances[np.arange(n_ts), assignments]) / n_ts
    
def _set_weights(w, n):
    """Return w if it is a valid weight vector of size n, and a vector of n 1s
    otherwise.
    """
    if w is None or len(w) != n:
        w = np.ones((n, ))
    return w

def _init_avg(X, barycenter_size):
    if X.shape[1] == barycenter_size:
        return np.nanmean(X, axis=0)
    else:
        X_avg = np.nanmean(X, axis=0)
        xnew = np.linspace(0, 1, barycenter_size)
        f = interp1d(np.linspace(0, 1, X_avg.shape[0]), X_avg,
                     kind="linear", axis=0)
        return f(xnew)

def _check_initial_guess(init, n_clusters):
    if hasattr(init, '__array__'):
        assert init.shape[0] == n_clusters, \
            "Initial guess index array must contain {} samples," \
            " {} given".format(n_clusters, init.shape[0])