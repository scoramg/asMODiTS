from sklearn.base import ClusterMixin, TransformerMixin, check_is_fitted
from sklearn.dummy import check_random_state

from DistanceMeasures.dtw import DTW
from DistanceMeasures.euclidean import Euclidean
import numpy as np
from sklearn.utils.extmath import stable_cumsum
import scipy.sparse as sp
from scipy.spatial.distance import cdist

from LearnMethods.utils import EmptyClusterError, _check_full_length, _check_initial_guess, _check_no_empty_cluster, _compute_inertia
from Utils.timeseries import to_time_series_dataset
from Utils.utils import check_dims
from LearnMethods.Clustering.bases import BaseModelPackage, TimeSeriesBaseEstimator, TimeSeriesCentroidBasedClusteringMixin


class TimeSeriesKMeans(TransformerMixin, ClusterMixin,
                       TimeSeriesCentroidBasedClusteringMixin,
                       BaseModelPackage, TimeSeriesBaseEstimator):
    """K-means clustering for time-series data.

    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.

    max_iter : int (default: 50)
        Maximum number of iterations of the k-means algorithm for a single run.

    tol : float (default: 1e-6)
        Inertia variation threshold. If at some point, inertia varies less than
        this threshold between two consecutive
        iterations, the model is considered to have converged and the algorithm
        stops.

    n_init : int (default: 1)
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of n_init
        consecutive runs in terms of inertia.

    metric : {"euclidean", "dtw", "softdtw"} (default: "euclidean")
        Metric to be used for both cluster assignment and barycenter
        computation. If "dtw", DBA is used for barycenter
        computation.

    max_iter_barycenter : int (default: 100)
        Number of iterations for the barycenter computation process. Only used
        if `metric="dtw"` or `metric="softdtw"`.

    metric_params : dict or None (default: None)
        Parameter values for the chosen metric.
        For metrics that accept parallelization of the cross-distance matrix
        computations, `n_jobs` key passed in `metric_params` is overridden by
        the `n_jobs` argument.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    dtw_inertia: bool (default: False)
        Whether to compute DTW inertia even if DTW is not the chosen metric.

    verbose : int (default: 0)
        If nonzero, print information about the inertia while learning
        the model and joblib progress messages are printed.  

    random_state : integer or numpy.RandomState, optional
        Generator used to initialize the centers. If an integer is given, it
        fixes the seed. Defaults to the global
        numpy random number generator.

    init : {'k-means++', 'random' or an ndarray} (default: 'k-means++')
        Method for initialization:
        'k-means++' : use k-means++ heuristic. See `scikit-learn's k_init_
        <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/\
        cluster/k_means_.py>`_ for more.
        'random': choose k observations (rows) at random from data for the
        initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, ts_size, d)
        and gives the initial centers.

    Attributes
    ----------
    labels_ : numpy.ndarray
        Labels of each point.

    cluster_centers_ : numpy.ndarray of shape (n_clusters, sz, d)
        Cluster centers.
        `sz` is the size of the time series used at fit time if the init method
        is 'k-means++' or 'random', and the size of the longest initial
        centroid if those are provided as a numpy array through init parameter.

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    n_iter_ : int
        The number of iterations performed during fit.

    Notes
    -----
        If `metric` is set to `"euclidean"`, the algorithm expects a dataset of
        equal-sized time series.

    Examples
    --------
    >>> from tslearn.generators import random_walks
    >>> X = random_walks(n_ts=50, sz=32, d=1)
    >>> km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5,
    ...                       random_state=0).fit(X)
    >>> km.cluster_centers_.shape
    (3, 32, 1)
    >>> km_dba = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=5,
    ...                           max_iter_barycenter=5,
    ...                           random_state=0).fit(X)
    >>> km_dba.cluster_centers_.shape
    (3, 32, 1)
    >>> km_sdtw = TimeSeriesKMeans(n_clusters=3, metric="softdtw", max_iter=5,
    ...                            max_iter_barycenter=5,
    ...                            metric_params={"gamma": .5},
    ...                            random_state=0).fit(X)
    >>> km_sdtw.cluster_centers_.shape
    (3, 32, 1)
    >>> X_bis = to_time_series_dataset([[1, 2, 3, 4],
    ...                                 [1, 2, 3],
    ...                                 [2, 5, 6, 7, 8, 9]])
    >>> km = TimeSeriesKMeans(n_clusters=2, max_iter=5,
    ...                       metric="dtw", random_state=0).fit(X_bis)
    >>> km.cluster_centers_.shape
    (2, 6, 1)
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if self.metric == "dtw":
            self.dtw = DTW(options=self.metric_params)
        if self.metric == "euclidean":
            self.euclidean = Euclidean(metric='euclidean', p=2)
    
    def _k_init_metric(self, X, cdist_metric, random_state, n_local_trials=None):
        """Init n_clusters seeds according to k-means++ with a custom distance
        metric.

        Parameters
        ----------
        X : array, shape (n_samples, n_timestamps, n_features)
            The data to pick seeds for.

        n_clusters : integer
            The number of seeds to choose

        cdist_metric : function
            Function to be called for cross-distance computations

        random_state : RandomState instance
            Generator used to initialize the centers.

        n_local_trials : integer, optional
            The number of seeding trials for each center (except the first),
            of which the one reducing inertia the most is greedily chosen.
            Set to None to make the number of trials depend logarithmically
            on the number of seeds (2+log(k)); this is the default.

        Notes
        -----
        Selects initial cluster centers for k-mean clustering in a smart way
        to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
        "k-means++: the advantages of careful seeding". ACM-SIAM symposium
        on Discrete algorithms. 2007

        Version adapted from scikit-learn for use with a custom metric in place of
        Euclidean distance.
        """
        n_samples, n_timestamps, n_features = X.shape

        centers = np.empty((self.n_clusters, n_timestamps, n_features),
                            dtype=X.dtype)

        # Set the number of local seeding trials if none is given
        if n_local_trials is None:
            # This is what Arthur/Vassilvitskii tried, but did not report
            # specific results for other than mentioning in the conclusion
            # that it helped.
            n_local_trials = 2 + int(np.log(self.n_clusters))

        # Pick first center randomly
        center_id = random_state.randint(n_samples)
        centers[0] = X[center_id]

        # Initialize list of closest distances and calculate current potential
        closest_dist_sq = cdist_metric(centers[0, np.newaxis], X) ** 2
        current_pot = closest_dist_sq.sum()

        # Pick the remaining n_clusters-1 points
        for c in range(1, self.n_clusters):
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            rand_vals = random_state.random_sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                            rand_vals)
            # XXX: numerical imprecision can result in a candidate_id out of range
            np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                    out=candidate_ids)

            # Compute distances to center candidates
            distance_to_candidates = cdist_metric(X[candidate_ids], X) ** 2

            # update closest distances squared and potential for each candidate
            np.minimum(closest_dist_sq, distance_to_candidates,
                        out=distance_to_candidates)
            candidates_pot = distance_to_candidates.sum(axis=1)

            # Decide which candidate is the best
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            # Permanently add best center candidate found in local tries
            centers[c] = X[best_candidate]

        return centers

    def _kmeans_plusplus(self,X, sample_weight, random_state, n_local_trials=None):
        """Computational component for initialization of n_clusters by
        k-means++. Prior validation of data is assumed.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The data to pick seeds for.

        n_clusters : int
            The number of seeds to choose.

        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in `X`.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared Euclidean norm of each data point.

        random_state : RandomState instance
            The generator used to initialize the centers.
            See :term:`Glossary <random_state>`.

        n_local_trials : int, default=None
            The number of seeding trials for each center (except the first),
            of which the one reducing inertia the most is greedily chosen.
            Set to None to make the number of trials depend logarithmically
            on the number of seeds (2+log(k)); this is the default.

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
            The initial centers for k-means.

        indices : ndarray of shape (n_clusters,)
            The index location of the chosen centers in the data array X. For a
            given index and center, X[index] = center.
        """
        n_samples, n_features = X.shape

        centers = np.empty((self.n_clusters, n_features), dtype=X.dtype)

        # Set the number of local seeding trials if none is given
        if n_local_trials is None:
            # This is what Arthur/Vassilvitskii tried, but did not report
            # specific results for other than mentioning in the conclusion
            # that it helped.
            n_local_trials = 2 + int(np.log(self.n_clusters))

        # Pick first center randomly and track index of point
        center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
        indices = np.full(self.n_clusters, -1, dtype=int)
        if sp.issparse(X):
            centers[0] = X[center_id].toarray()
        else:
            centers[0] = X[center_id]
        indices[0] = center_id

        # Initialize list of closest distances and calculate current potential
        #closest_dist_sq = _euclidean_distances(
        #    centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
        #)
        closest_dist_sq = self.euclidean.distance(dataset1=centers[0, np.newaxis], dataset2=X)
        current_pot = closest_dist_sq @ sample_weight

        # Pick the remaining n_clusters-1 points
        for c in range(1, self.n_clusters):
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            rand_vals = random_state.uniform(size=n_local_trials) * current_pot
            candidate_ids = np.searchsorted(
                stable_cumsum(sample_weight * closest_dist_sq), rand_vals
            )
            # XXX: numerical imprecision can result in a candidate_id out of range
            np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

            # Compute distances to center candidates
            #distance_to_candidates = _euclidean_distances(
            #    X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
            #)
            distance_to_candidates = self.euclidean.distance(dataset1=X[candidate_ids], dataset2=X)

            # update closest distances squared and potential for each candidate
            np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
            candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

            # Decide which candidate is the best
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            # Permanently add best center candidate found in local tries
            if sp.issparse(X):
                centers[c] = X[best_candidate].toarray()
            else:
                centers[c] = X[best_candidate]
            indices[c] = best_candidate

        return centers, indices
    
    def _is_fitted(self):
        check_is_fitted(self, ['cluster_centers_'])
        return True

    """ def _get_metric_params(self):
        if not hasattr(self, "metric_params"):
            metric_params = {}
        else:
            metric_params = self.metric_params.copy()
        if "n_jobs" in metric_params.keys():
            del metric_params["n_jobs"]
        return metric_params """

    def _fit_one_init(self, X, rs):
        n_ts, sz, d = X.shape
        if hasattr(self.init, '__array__'):
            self.cluster_centers_ = self.init.copy()
        elif isinstance(self.init, str) and self.init == "k-means++":
            if self.metric == "euclidean":
                self.cluster_centers_ = self._kmeans_plusplus(
                    X.reshape((n_ts, -1)),
                    self.n_clusters,
                    random_state=rs
                )[0].reshape((-1, sz, d))
            else:
                if self.metric == "dtw":
                    def metric_fun(x, y):
                        return self.dtw.cdist(x, y, n_jobs=self.n_jobs, verbose=self.verbose)
                else:
                    raise ValueError(
                        "Incorrect metric: %s (should be one of 'dtw', "
                        "'softdtw', 'euclidean')" % self.metric
                    )
                self.cluster_centers_ = self._k_init_metric(X, cdist_metric=metric_fun,
                                                       random_state=rs)
        elif self.init == "random":
            indices = rs.choice(X.shape[0], self.n_clusters)
            self.cluster_centers_ = X[indices].copy()
        else:
            raise ValueError("Value %r for parameter 'init'"
                             "is invalid" % self.init)
        self.cluster_centers_ = _check_full_length(self.cluster_centers_)
        old_inertia = np.inf

        for it in range(self.max_iter):
            self._assign(X)
            if self.verbose:
                print("%.3f" % self.inertia_, end=" --> ")
            self._update_centroids(X)

            if np.abs(old_inertia - self.inertia_) < self.tol:
                break
            old_inertia = self.inertia_
        if self.verbose:
            print("")

        self._iter = it + 1

        return self

    def _transform(self, X):
        if self.metric == "euclidean":
            return self.euclidean.cdist(X=X.reshape((X.shape[0], -1)),
                         cluster_centers=self.cluster_centers_.reshape((self.n_clusters, -1)))
        elif self.metric == "dtw":
            return self.dtw.cdist(X, self.cluster_centers_, n_jobs=self.n_jobs, verbose=self.verbose)
            
        else:
            raise ValueError("Incorrect metric: %s (should be one of 'dtw', "
                             "'softdtw', 'euclidean')" % self.metric)

    def _assign(self, X, update_class_attributes=True):
        dists = self._transform(X)
        matched_labels = dists.argmin(axis=1)
        if update_class_attributes:
            self.labels_ = matched_labels
            _check_no_empty_cluster(self.labels_, self.n_clusters)
            if self.dtw_inertia and self.metric != "dtw":
                inertia_dists = self.dtw.cdist(X, self.cluster_centers_, n_jobs=self.n_jobs, verbose=self.verbose)
                
            else:
                inertia_dists = dists
            self.inertia_ = _compute_inertia(inertia_dists,
                                             self.labels_,
                                             self._squared_inertia)
        return matched_labels

    def _update_centroids(self, X):
        for k in range(self.n_clusters):
            if self.metric == "dtw":
                self.cluster_centers_[k] = self.dtw.barycenter_averaging(
                    X=X[self.labels_ == k],
                    init_barycenter=self.cluster_centers_[k])
                
            else:
                self.cluster_centers_[k] = self.euclidean.barycenter(
                    X=X[self.labels_ == k])

    def fit(self, X, y=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        y
            Ignored
        """
        
       
        if hasattr(self.init, '__array__'):
            X = check_dims(X, X_fit_dims=self.init.shape,
                           extend=True,
                           check_n_features_only=(self.metric != "euclidean"))

        self.labels_ = None
        self.inertia_ = np.inf
        self.cluster_centers_ = None
        self._X_fit = None
        self._squared_inertia = True

        self.n_iter_ = 0

        max_attempts = max(self.n_init, 10)

        X_ = to_time_series_dataset(X)
        rs = check_random_state(self.random_state)

        if isinstance(self.init, str) and self.init == "k-means++" and \
                        self.metric == "euclidean":
            n_ts, sz, d = X_.shape
            x_squared_norms = self.euclidean.cdist(X = X_.reshape((n_ts, -1)),
                                    cluster_centers=np.zeros((1, sz * d))).reshape((1, -1))
        else:
            x_squared_norms = None
        _check_initial_guess(self.init, self.n_clusters)

        best_correct_centroids = None
        min_inertia = np.inf
        n_successful = 0
        n_attempts = 0
        while n_successful < self.n_init and n_attempts < max_attempts:
            try:
                if self.verbose and self.n_init > 1:
                    print("Init %d" % (n_successful + 1))
                n_attempts += 1
                self._fit_one_init(X_, rs)
                if self.inertia_ < min_inertia:
                    best_correct_centroids = self.cluster_centers_.copy()
                    min_inertia = self.inertia_
                    self.n_iter_ = self._iter
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")
        self._post_fit(X_, best_correct_centroids, min_inertia)
        return self

    def fit_predict(self, X, y=None):
        """Fit k-means clustering using X and then predict the closest cluster
        each time series in X belongs to.

        It is more efficient to use this method than to sequentially call fit
        and predict.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        y
            Ignored

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        #X = check_array(X, allow_nd=True, force_all_finite='allow-nan')
        return self.fit(X, y).labels_

    def predict(self, X):
        """Predict the closest cluster each time series in X belongs to.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, 'cluster_centers_')
        X = check_dims(X, X_fit_dims=self.cluster_centers_.shape,
                       extend=True,
                       check_n_features_only=(self.metric != "euclidean"))
        return self._assign(X, update_class_attributes=False)

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster 
        centers.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset

        Returns
        -------
        distances : array of shape=(n_ts, n_clusters)
            Distances to cluster centers
        """
        check_is_fitted(self, 'cluster_centers_')
        X = check_dims(X, X_fit_dims=self.cluster_centers_.shape,
                       extend=True,
                       check_n_features_only=(self.metric != "euclidean"))
        return self._transform(X)

    def _more_tags(self):
        return {'allow_nan': True, 'allow_variable_length': True}