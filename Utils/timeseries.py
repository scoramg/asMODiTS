from abc import ABCMeta, abstractmethod
import json
from logging import warn
import pickle
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
try:
    from sklearn.utils.estimator_checks import _NotAnArray as NotAnArray
except ImportError:  # Old sklearn versions
    from sklearn.utils.estimator_checks import NotAnArray

from scipy.interpolate import interp1d
    
class TimeSeriesResampler(TransformerMixin):
    """Resampler for time series. Resample time series so that they reach the
    target size.

    Parameters
    ----------
    sz : int
        Size of the output time series.

    Examples
    --------
    >>> TimeSeriesResampler(sz=5).fit_transform([[0, 3, 6]])
    array([[[0. ],
            [1.5],
            [3. ],
            [4.5],
            [6. ]]])
    """
    def __init__(self, sz):
        self.sz_ = sz

    def fit(self, X, y=None, **kwargs):
        """A dummy method such that it complies to the sklearn requirements.
        Since this method is completely stateless, it just returns itself.

        Parameters
        ----------
        X
            Ignored

        Returns
        -------
        self
        """
        return self

    def _transform_unit_sz(self, X):
        n_ts, sz, d = X.shape
        X_out = np.empty((n_ts, self.sz_, d))
        for i in range(X.shape[0]):
            X_out[i] = np.nanmean(X[i], axis=0, keepdims=True)
        return X_out

    def fit_transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be resampled.

        Returns
        -------
        numpy.ndarray
            Resampled time series dataset.
        """
        return self.fit(X).transform(X)

    def transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be resampled.

        Returns
        -------
        numpy.ndarray
            Resampled time series dataset.
        """
        X_ = to_time_series_dataset(X)
        if self.sz_ == 1:
            return self._transform_unit_sz(X_)
        n_ts, sz, d = X_.shape
        equal_size = check_equal_size(X_)
        X_out = np.empty((n_ts, self.sz_, d))
        for i in range(X_.shape[0]):
            xnew = np.linspace(0, 1, self.sz_)
            if not equal_size:
                sz = ts_size(X_[i])
            for di in range(d):
                f = interp1d(np.linspace(0, 1, sz), X_[i, :sz, di],
                             kind="slinear")
                X_out[i, :, di] = f(xnew)
        return X_out

def ts_size(ts):
    """Returns actual time series size.

    Final timesteps that have `NaN` values for all dimensions will be removed
    from the count. Infinity and negative infinity ar considered valid time
    series values.

    Parameters
    ----------
    ts : array-like
        A time series.

    Returns
    -------
    int
        Actual size of the time series.

    Examples
    --------
    >>> ts_size([1, 2, 3, numpy.nan])
    3
    >>> ts_size([1, numpy.nan])
    1
    >>> ts_size([numpy.nan])
    0
    >>> ts_size([[1, 2],
    ...          [2, 3],
    ...          [3, 4],
    ...          [numpy.nan, 2],
    ...          [numpy.nan, numpy.nan]])
    4
    >>> ts_size([numpy.nan, 3, numpy.inf, numpy.nan])
    3
    """
    ts_ = to_time_series(ts)
    sz = ts_.shape[0]
    #print(ts_[sz - 1][0])
    while sz > 0 and np.all(np.isnan(ts_[sz - 1][0])):
        sz -= 1
    return sz
    
def to_time_series(ts, remove_nans=False):
    """Transforms a time series so that it fits the format used in ``tslearn``
    models.

    Parameters
    ----------
    ts : array-like
        The time series to be transformed.
    remove_nans : bool (default: False)
        Whether trailing NaNs at the end of the time series should be removed
        or not

    Returns
    -------
    numpy.ndarray of shape (sz, d)
        The transformed time series. This is always guaraneteed to be a new
        time series and never just a view into the old one.

    Examples
    --------
    >>> to_time_series([1, 2])
    array([[1.],
           [2.]])
    >>> to_time_series([1, 2, numpy.nan])
    array([[ 1.],
           [ 2.],
           [nan]])
    >>> to_time_series([1, 2, numpy.nan], remove_nans=True)
    array([[1.],
           [2.]])

    See Also
    --------
    to_time_series_dataset : Transforms a dataset of time series
    """
    #print(ts[0])
    ts_out = np.array(ts[0], copy=True)
    if ts_out.ndim <= 1:
        ts_out = ts_out.reshape((-1, 1))
    if ts_out.dtype != float:
        ts_out = ts_out.astype(float)
    if remove_nans:
        ts_out = ts_out[:ts_size(ts_out)]
    return ts_out

def to_time_series_dataset(dataset, dtype=float):
    """Transforms a time series dataset so that it fits the format used in
    ``tslearn`` models.

    Parameters
    ----------
    dataset : array-like
        The dataset of time series to be transformed. A single time series will
        be automatically wrapped into a dataset with a single entry.
    dtype : data type (default: float)
        Data type for the returned dataset.

    Returns
    -------
    numpy.ndarray of shape (n_ts, sz, d)
        The transformed dataset of time series.

    Examples
    --------
    >>> to_time_series_dataset([[1, 2]])
    array([[[1.],
            [2.]]])
    >>> to_time_series_dataset([1, 2])
    array([[[1.],
            [2.]]])
    >>> to_time_series_dataset([[1, 2], [1, 4, 3]])
    array([[[ 1.],
            [ 2.],
            [nan]],
    <BLANKLINE>
           [[ 1.],
            [ 4.],
            [ 3.]]])
    >>> to_time_series_dataset([]).shape
    (0, 0, 0)

    See Also
    --------
    to_time_series : Transforms a single time series
    """
    try:
        import pandas as pd
        if isinstance(dataset, pd.DataFrame):
            return to_time_series_dataset(np.array(dataset))
    except ImportError:
        pass
    if isinstance(dataset, NotAnArray):  # Patch to pass sklearn tests
        return to_time_series_dataset(np.array(dataset))
    if len(dataset) == 0:
        return np.zeros((0, 0, 0))
    #print(np.array(dataset))
    if np.array(dataset[0]).ndim == 0:
        dataset = [dataset]
    n_ts = len(dataset)
    max_sz = max([ts_size(to_time_series(ts, remove_nans=True))
                  for ts in dataset])
    d = to_time_series(dataset[0]).shape[1]
    dataset_out = np.zeros((n_ts, max_sz, d), dtype=dtype) + np.nan
    for i in range(n_ts):
        ts = to_time_series(dataset[i], remove_nans=True)
        dataset_out[i, :ts.shape[0]] = ts
    return dataset_out.astype(dtype)

def time_series_to_lists(timeseries):
    dataset_out = []
    #print(type(timeseries))
    dataset_out = [ele.tolist() for ele in timeseries]
    return dataset_out
    #n_ts = len(timeseries)
    #dataset_out = []
    #for i in range(n_ts):
    #    dataset_out.append(timeseries[i])
    #return np.array(dataset_out,ndmin=2)

def check_equal_size(dataset):
    """Check if all time series in the dataset have the same size.

    Parameters
    ----------
    dataset: array-like
        The dataset to check.

    Returns
    -------
    bool
        Whether all time series in the dataset have the same size.

    Examples
    --------
    >>> check_equal_size([[1, 2, 3], [4, 5, 6], [5, 3, 2]])
    True
    >>> check_equal_size([[1, 2, 3, 4], [4, 5, 6], [5, 3, 2]])
    False
    >>> check_equal_size([])
    True
    """
    dataset_ = to_time_series_dataset(dataset)
    if len(dataset_) == 0:
        return True

    size = ts_size(dataset[0])
    return all(ts_size(ds) == size for ds in dataset_[1:])
