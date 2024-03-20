import numpy as np
from scipy.spatial.distance import squareform, cdist as _cdist, pdist

from LearnMethods.utils import _set_weights
from Utils.timeseries import to_time_series_dataset

class Euclidean:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def distance(self, dataset1, dataset2=None):
        if dataset2 is not None:
            return _cdist(dataset1, dataset2, metric=self.metric, p=self.p)
        else:
            return squareform(pdist(dataset1, metric=self.metric, p=self.p))
        
    def barycenter(self, X, weights=None):
        """Standard Euclidean barycenter computed from a set of time series.

        Parameters
        ----------
        X : array-like, shape=(n_ts, sz, d)
            Time series dataset.

        weights: None or array
            Weights of each X[i]. Must be the same size as len(X).
            If None, uniform weights are used.

        Returns
        -------
        numpy.array of shape (sz, d)
            Barycenter of the provided time series dataset.

        Notes
        -----
            This method requires a dataset of equal-sized time series

        Examples
        --------
        >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
        >>> bar = euclidean_barycenter(time_series)
        >>> bar.shape
        (4, 1)
        >>> bar
        array([[1. ],
            [2. ],
            [3.5],
            [4.5]])
        """
        X_ = to_time_series_dataset(X)
        weights = _set_weights(weights, X_.shape[0])
        return np.average(X_, axis=0, weights=weights)
    
    def cdist(self, X, cluster_centers):
        _cdist(X.reshape((X.shape[0], -1)),
                         cluster_centers,
                         metric=self.metric, p=self.p)