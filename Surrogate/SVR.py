import argparse
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.base import RegressorMixin
from sklearn.svm import SVR as sklearn_svr
from sklearn.utils.validation import check_is_fitted
from Surrogate.ModelBase import ModelBase
from DistanceMeasures.gak import GAK, VARIABLE_LENGTH_METRICS
from Utils.timeseries import to_time_series_dataset
from Utils.utils import check_dims
from Utils.parser import Parser
import Settings as conf
from argparse import Namespace
from hyperopt import hp

class Optimizer:
    def __init__(self):
        pass
    
    def to_namespace(self, params):
        args = {
            "name": 'SVR',
            "ue": "archive",
            "gu": 5,
            "dist_metric": "gak",
            "dtw_sakoechiba_w": params["svr_dtw_sakoechiba_w"],
            "svr_c": params["svr_c"],
            "svr_gamma": 0,
            "train_rep": params["svr_train_rep"],
        }
        
        return Namespace(**args)
    
    def space(self, label):
        space = {
            label: 'svr', 
            'svr_c': hp.quniform('svr_c', 1, 10, 1),
            'svr_dtw_sakoechiba_w': hp.uniform('svr_dtw_sakoechiba_w', 0.0, 1.0),
            'svr_train_rep': hp.choice('svr_train_rep', ['allnorm'])
        }
        return space
    
class Parameters(Parser):
        
    def __init__(self, **kwargs):
        kwargs['add_help'] = False
        super().__init__(**kwargs)
        
        self.title = "SVR"
        self.prog = "svr"
        self.description = "Support Vector Regression model"
        
        self.add_argument("--name", help="Name of the model. Default value = %(default)s", default=self.title)
        self.add_argument("--gu", type=int, help="The generations' number for evaluating the entire population in the original models (generation strategy). Type of data: integer. Required argument. Default value = %(default)s",default=5)
        self.add_argument("--train-rep", type=str, help="Representation type used for the surrogate model train set. 'all' = A Vector with all values, 'allnorm' = A normalized vector with all values, 'numcuts' = Vector with only number of cuts, 'stats' = Vector with stats values, 'cutdits' = Vector with cut distributions. Type of data: string. Default value = %(default)s",default="all")
        self.add_argument("--dist-metric", type=str, choices=SVR.distances_allowed,help="List of distance metric: {dists}. Type of data: string. Default value = %(default)s".format(dists=SVR.distances_allowed),default=SVR.distances_allowed[0])
        self.add_argument("--dtw-sakoechiba-w", type=float, help="Window size parameter for sakoechiba dtw constraint. Type of data: double. Default value = %(default)s", default=0.1)        
        self.add_argument("--svr-c", type=int, help="Penalty parameter C of the error term. Type of data: int. Default value = %(default)s", default=1)
        self.add_argument("--svr-gamma", type=float, help="Kernel coefficient for 'gak', 'rbf', 'poly' and 'sigmoid'. Type of data: float. Default value = %(default)s (in sklearn.SVR it is equal to 'auto')", default=0)
    
    def handle(self, arguments):
        print(arguments)


class SVR(RegressorMixin, ModelBase):
    
    distances_allowed = conf.DISTANCES["SVR"]
    
    def __init__(self, id_model, options=None):
        super().__init__(id_model=id_model, options=options)
        self.options = options
        
        self.n_jobs = None
        self.verbose = 0
        self.degree = 3
        self.coef0=0.0
        self.tol=0.001
        self.epsilon=0.1
        self.shrinking=True
        self.cache_size=200
        self.max_iter=-1
        self.gak = GAK(n_jobs=self.n_jobs, verbose=self.verbose)
        self.gamma = None
        
    def preprocess_sklearn(self, X, y=None, fit_time=False):
        
        X = to_time_series_dataset(X)
        
        if fit_time:
            self._X_fit = X
            if self.options.model[self.id_model].svr_gamma == 0:
                self.gamma = self.gak.gamma_soft_dtw(X)
            else:
                self.gamma = self.options.model[self.id_model].svr_gamma
            self.classes_ = np.unique(y)
        else:
            check_is_fitted(self, ['svr_estimator_', '_X_fit'])
            X = check_dims(
                X,
                X_fit_dims=self._X_fit.shape,
                extend=True,
                check_n_features_only=(self.options.model[self.id_model].dist_metric in VARIABLE_LENGTH_METRICS)
            )

        if self.options.model[self.id_model].dist_metric in VARIABLE_LENGTH_METRICS:
            assert self.options.model[self.id_model].dist_metric == "gak"
            self.estimator_kernel_ = "precomputed"
            if fit_time:
                sklearn_X, matrix_distances = self.gak.cdist_gak(X,
                                      sigma=np.sqrt(self.gamma / 2.))
            else:
                sklearn_X, matrix_distances = self.gak.cdist_gak(X,
                                      self._X_fit,
                                      sigma=np.sqrt(self.gamma / 2.))
        else:
            self.estimator_kernel_ = self.options.model[self.id_model].dist_metric
            sklearn_X = self.gak.to_sklearn_dataset(X)

        if y is None:
            return sklearn_X, matrix_distances
        else:
            return sklearn_X, y, matrix_distances
    
    @property
    def n_iter_(self):
        warnings.warn('n_iter_ is always set to 1 for TimeSeriesSVR, since '
                      'it is non-trivial to access the underlying libsvm')
        return 1
    
    @property
    def support_vectors_(self):
        check_is_fitted(self, '_X_fit')
        return self._X_fit[self.svr_estimator_.support_]
    
    def get_name(self):
        return self.class_name+self.options.model[self.id_model].dist_metric.upper()+"("+str(self.options.model[self.id_model].gu)+"_"+str(int(self.options.model[self.id_model].dtw_sakoechiba_w*100))+"_"+str(self.options.model[self.id_model].svr_c)+"_"+str(self.options.model[self.id_model].svr_gamma)+")"
    
    
    def train(self):
        xtrain, classes = self.training_set.to_train_set(self.id_model)
        ytrain = classes[:,self.id_model]
        self.training_number += 1
        self.is_trained = True
        
        self.xtrain, self.ytrain, _ = self.preprocess_sklearn(xtrain, ytrain, fit_time=True)
        self.svr_estimator_ = sklearn_svr(
            C=self.options.model[self.id_model].svr_c, kernel=self.estimator_kernel_, degree=self.degree,
            gamma=self.gamma, coef0=self.coef0, shrinking=self.shrinking,
            tol=self.tol, cache_size=self.cache_size,
            verbose=self.verbose, max_iter=self.max_iter
        )
        sample_weight = None
        self.svr_estimator_.fit(self.xtrain, self.ytrain, sample_weight=sample_weight)
    
    def predict(self, x):
        
        self.normalized_matrix_distance, self.matrix_distances = self.preprocess_sklearn(x, fit_time=False)
        self.matrix_distances = -self.matrix_distances
            
        return self.svr_estimator_.predict(self.normalized_matrix_distance)
    
    def classify(self, X_test, y_test):
        y_pred = self.predict_set(X_test)
        error_rate = mean_squared_error(y_test, y_pred)
        return  error_rate
    
    def export_matlab(self):
        data = {}
        data["id_model"] = self.id_model
        data["name"] = self.class_name
        data["training_set"] = self.training_set.export_matlab(isAccumulated=False)
        data["training_number"] = self.training_number
        data["archive"] = self.archive.export_matlab()
        return data
    
    def copy(self):
        svr = SVR(id_model=self.id_model, options=self.options)
        svr.x = self.xtrain.copy() if self.xtrain else None
        svr.y = self.ytrain.copy() if self.ytrain else None
        svr.svr_estimator_ = self.svr_estimator_ if hasattr(self, 'svr_estimator_') else None
        svr._X_fit = self._X_fit if hasattr(self, '_X_fit') else None
        svr.gamma = self.gamma if self.gamma else None
        svr.gak = self.gak
        return svr