from numba import njit
from joblib import Parallel, delayed
try:
    from sklearn.utils.estimator_checks import _NotAnArray as NotAnArray
except ImportError:  # Old sklearn versions
    from sklearn.utils.estimator_checks import NotAnArray
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
import numpy as np
from sklearn.metrics import mean_squared_error
from DistanceMeasures.utils import cdist_generic
from Utils.timeseries import to_time_series_dataset, to_time_series, check_equal_size
from scipy.spatial.distance import cdist, pdist

VARIABLE_LENGTH_METRICS = ["ctw", "dtw", "gak", "sax", "softdtw", "lcss"]

@njit(nogil=True)
def njit_gak(s1, s2, gram):
    l1 = s1.shape[0]
    l2 = s2.shape[0]

    cum_sum = np.zeros((l1 + 1, l2 + 1))
    cum_sum[0, 0] = 1.0

    for i in range(l1):
        for j in range(l2):
    #for i,j in list(itertools.product(np.arange(0,l1),np.arange(0,l2))):
            cum_sum[i + 1, j + 1] = (
                cum_sum[i, j + 1] + cum_sum[i + 1, j] + cum_sum[i, j]
            ) * gram[i, j]

    return cum_sum[l1, l2]

class GAK:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        #self.n_jobs=n_jobs
        #self.verbose=verbose
        

    def _gak_gram(self, s1, s2, sigma):
        gram = -cdist(s1, s2, "sqeuclidean") / (2 * sigma**2)
        gram -= np.log(2 - np.exp(gram))
        return np.exp(gram)

    def unnormalized_gak(self, s1, s2, sigma=1.0):
        r"""Compute Global Alignment Kernel (GAK) between (possibly
        multidimensional) time series and return it.

        It is not required that both time series share the same size, but they must
        be the same dimension. GAK was
        originally presented in [1]_.
        This is an unnormalized version.

        Parameters
        ----------
        s1
            A time series
        s2
            Another time series
        sigma : float (default 1.)
            Bandwidth of the internal gaussian kernel used for GAK

        Returns
        -------
        float
            Kernel value

        Examples
        --------
        >>> unnormalized_gak([1, 2, 3],
        ...                  [1., 2., 2., 3.],
        ...                  sigma=2.)  # doctest: +ELLIPSIS
        15.358...
        >>> unnormalized_gak([1, 2, 3],
        ...                  [1., 2., 2., 3., 4.])  # doctest: +ELLIPSIS
        3.166...

        See Also
        --------
        gak : normalized version of GAK that ensures that k(x,x) = 1
        cdist_gak : Compute cross-similarity matrix using Global Alignment kernel

        References
        ----------
        .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
        """
        s1 = to_time_series(s1, remove_nans=True) #Aquí puede estar dando lo duplicado
        s2 = to_time_series(s2, remove_nans=True)

        gram = self._gak_gram(s1, s2, sigma=sigma)

        gak_val = njit_gak(s1, s2, gram)
        return gak_val
    
    def unnormalized_gak2(self, s1, s2, sigma=1.0):
        s1 = np.array(s1)
        s1 = np.expand_dims(s1, axis=1)
        s2 = np.array(s2)
        s2 = np.expand_dims(s2, axis=1)

        gram = self._gak_gram(s1, s2, sigma=sigma)

        gak_val = njit_gak(s1, s2, gram)
        return gak_val
    
    def distance(self,dataset1, dataset2=None):
        sigma = self.sigma_gak(dataset=dataset1)
        unnormalized_matrix = cdist_generic(
            dist_fun=self.unnormalized_gak2,
            dataset1=dataset1,
            dataset2=dataset2,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            sigma=sigma,
            compute_diagonal=True,
        )
        #print('Bef: ', np.min(unnormalized_matrix),np.max(unnormalized_matrix), np.mean(unnormalized_matrix))
        #dataset1 = to_time_series_dataset(dataset1)
        if dataset2 is None:
            diagonal = np.diag(np.sqrt(1.0 / np.diag(unnormalized_matrix)))
            diagonal_left = diagonal_right = diagonal
        else:
            #dataset2 = to_time_series_dataset(dataset2)
            diagonal_left = Parallel(n_jobs=self.n_jobs, prefer="threads", verbose=self.verbose)(
                delayed(self.unnormalized_gak2)(dataset1[i], dataset1[i], sigma=sigma)
                for i in range(len(dataset1))
            )
            #diagonal_left = [self.unnormalized_gak2(dataset1[i], dataset1[i], sigma=sigma) for i in range(len(dataset1))]
            diagonal_right = Parallel(n_jobs=self.n_jobs, prefer="threads", verbose=self.verbose)(
                delayed(self.unnormalized_gak2)(dataset2[j], dataset2[j], sigma=sigma)
                for j in range(len(dataset2))
            )
            #diagonal_right = [self.unnormalized_gak2(dataset2[j], dataset2[j], sigma=sigma) for j in range(len(dataset2))]
            diagonal_left = np.diag(1.0 / np.sqrt(diagonal_left))
            diagonal_right = np.diag(1.0 / np.sqrt(diagonal_right))
        return (diagonal_left.dot(unnormalized_matrix)).dot(diagonal_right)
        

    def cdist_gak(self, dataset1, dataset2=None, sigma=1.0):
        r"""Compute cross-similarity matrix using Global Alignment kernel (GAK).

        GAK was originally presented in [1]_.

        Parameters
        ----------
        dataset1
            A dataset of time series
        dataset2
            Another dataset of time series
        sigma : float (default 1.)
            Bandwidth of the internal gaussian kernel used for GAK
        n_jobs : int or None, optional (default=None)
            The number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See scikit-learns'
            `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`__
            for more details.
        verbose : int, optional (default=0)
            The verbosity level: if non zero, progress messages are printed.
            Above 50, the output is sent to stdout.
            The frequency of the messages increases with the verbosity level.
            If it more than 10, all iterations are reported.
            `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
            for more details.

        Returns
        -------
        numpy.ndarray
            Cross-similarity matrix

        Examples
        --------
        >>> cdist_gak([[1, 2, 2, 3], [1., 2., 3., 4.]], sigma=2.)
        array([[1.        , 0.65629661],
            [0.65629661, 1.        ]])
        >>> cdist_gak([[1, 2, 2], [1., 2., 3., 4.]],
        ...           [[1, 2, 2, 3], [1., 2., 3., 4.], [1, 2, 2, 3]],
        ...           sigma=2.)
        array([[0.71059484, 0.29722877, 0.71059484],
            [0.65629661, 1.        , 0.65629661]])

        See Also
        --------
        gak : Compute Global Alignment kernel

        References
        ----------
        .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
        """  # noqa: E501
        unnormalized_matrix = cdist_generic(
            dist_fun=self.unnormalized_gak,
            dataset1=dataset1,
            dataset2=dataset2,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            sigma=sigma,
            compute_diagonal=True,
        )
        #print('Bef: ', np.min(unnormalized_matrix),np.max(unnormalized_matrix), np.mean(unnormalized_matrix))
        #dataset1 = to_time_series_dataset(dataset1)
        if dataset2 is None:
            diagonal = np.diag(np.sqrt(1.0 / np.diag(unnormalized_matrix)))
            diagonal_left = diagonal_right = diagonal
        else:
            #dataset2 = to_time_series_dataset(dataset2)
            diagonal_left = Parallel(n_jobs=self.n_jobs, prefer="threads", verbose=self.verbose)(
                delayed(self.unnormalized_gak)(dataset1[i], dataset1[i], sigma=sigma)
                for i in range(len(dataset1))
            )
            diagonal_right = Parallel(n_jobs=self.n_jobs, prefer="threads", verbose=self.verbose)(
                delayed(self.unnormalized_gak)(dataset2[j], dataset2[j], sigma=sigma)
                for j in range(len(dataset2))
            )
            diagonal_left = np.diag(1.0 / np.sqrt(diagonal_left))
            diagonal_right = np.diag(1.0 / np.sqrt(diagonal_right))
        return (diagonal_left.dot(unnormalized_matrix)).dot(diagonal_right), unnormalized_matrix

    def sigma_gak(self, dataset, n_samples=100, random_state=None):
        r"""Compute sigma value to be used for GAK.

        This method was originally presented in [1]_.

        Parameters
        ----------
        dataset
            A dataset of time series
        n_samples : int (default: 100)
            Number of samples on which median distance should be estimated
        random_state : integer or numpy.RandomState or None (default: None)
            The generator used to draw the samples. If an integer is given, it
            fixes the seed. Defaults to the global numpy random number generator.

        Returns
        -------
        float
            Suggested bandwidth (:math:`\sigma`) for the Global Alignment kernel

        Examples
        --------
        >>> dataset = [[1, 2, 2, 3], [1., 2., 3., 4.]]
        >>> sigma_gak(dataset=dataset,
        ...           n_samples=200,
        ...           random_state=0)  # doctest: +ELLIPSIS
        2.0...

        See Also
        --------
        gak : Compute Global Alignment kernel
        cdist_gak : Compute cross-similarity matrix using Global Alignment kernel

        References
        ----------
        .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
        """
        random_state = check_random_state(random_state)
        dataset = to_time_series_dataset(dataset=dataset)
        n_ts, sz, d = dataset.shape
        if check_equal_size(dataset):
            sz = np.min([ts.size for ts in dataset])
        if n_ts * sz < n_samples:
            replace = True
        else:
            replace = False
        sample_indices = random_state.choice(n_ts * sz, size=n_samples, replace=replace)
        dists = pdist(dataset[:, :sz, :].reshape((-1, d))[sample_indices], metric="euclidean")
        if (np.median(dists) * np.sqrt(sz)) == 0:
            print("[n_ts, sz, d]: ",dataset.shape,", dists: ", dists, ", np.median(dists): ", np.median(dists))
            return 1.0
        else:
            return np.median(dists) * np.sqrt(sz)

    def gamma_soft_dtw(self, dataset, n_samples=100, random_state=None):
        r"""Compute gamma value to be used for GAK/Soft-DTW.

        This method was originally presented in [1]_.

        Parameters
        ----------
        dataset
            A dataset of time series
        n_samples : int (default: 100)
            Number of samples on which median distance should be estimated
        random_state : integer or numpy.RandomState or None (default: None)
            The generator used to draw the samples. If an integer is given, it
            fixes the seed. Defaults to the global numpy random number generator.

        Returns
        -------
        float
            Suggested :math:`\gamma` parameter for the Soft-DTW

        Examples
        --------
        >>> dataset = [[1, 2, 2, 3], [1., 2., 3., 4.]]
        >>> gamma_soft_dtw(dataset=dataset,
        ...                n_samples=200,
        ...                random_state=0)  # doctest: +ELLIPSIS
        8.0...

        See Also
        --------
        sigma_gak : Compute sigma parameter for Global Alignment kernel

        References
        ----------
        .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
        """
        return 2. * self.sigma_gak(dataset=dataset,
                            n_samples=n_samples,
                            random_state=random_state) ** 2

    def to_sklearn_dataset(self, dataset, dtype=float, return_dim=False):
        """Transforms a time series dataset so that it fits the format used in
        ``sklearn`` estimators.

        Parameters
        ----------
        dataset : array-like
            The dataset of time series to be transformed.
        dtype : data type (default: float64)
            Data type for the returned dataset.
        return_dim : boolean  (optional, default: False)
            Whether the dimensionality (third dimension should be returned together
            with the transformed dataset).

        Returns
        -------
        numpy.ndarray of shape (n_ts, sz * d)
            The transformed dataset of time series.
        int (optional, if return_dim=True)
            The dimensionality of the original tslearn dataset (third dimension)

        Examples
        --------
        >>> to_sklearn_dataset([[1, 2]], return_dim=True)
        (array([[1., 2.]]), 1)
        >>> to_sklearn_dataset([[1, 2], [1, 4, 3]])
        array([[ 1.,  2., nan],
            [ 1.,  4.,  3.]])

        See Also
        --------
        to_time_series_dataset : Transforms a time series dataset to ``tslearn``
        format.
        """
        tslearn_dataset = to_time_series_dataset(dataset, dtype=dtype)
        n_ts = tslearn_dataset.shape[0]
        d = tslearn_dataset.shape[2]
        if return_dim:
            return tslearn_dataset.reshape((n_ts, -1)), d
        else:
            return tslearn_dataset.reshape((n_ts, -1))