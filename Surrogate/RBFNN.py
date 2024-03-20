from sklearn.metrics import mean_squared_error
import numpy as np

from Utils.timeseries import to_time_series_dataset
from tslearn.clustering.kmeans import TimeSeriesKMeans

from Surrogate.ModelBase import ModelBase

from Utils.parser import Parser
import Settings as conf
from argparse import Namespace
from hyperopt import hp

class Optimizer:
    def __init__(self):
        pass
    
    def to_namespace(self, params):
        args = {
            "name": 'RBFNN',
            "ue": "archive",
            "gu": 5,
            "dist_metric": "dtw",
            "dtw_sakoechiba_w": params["rbfnn_dtw_sakoechiba_w"],
            "rbfnn_func": "gaussian",
            "rbfnn_epsilon": 0,
            "rbfnn_k": int(params["rbfnn_k"]),
            "rbfnn_epochs": params["rbfnn_epochs"],
            "rbfnn_lr": params["rbfnn_lr"],
            "train_rep": "allnorm"
        }
        
        return Namespace(**args)
    
    def space(self, label):
        space = {
            label: 'rbfnn', 
            'rbfnn_dtw_sakoechiba_w': hp.uniform('rbfnn_dtw_sakoechiba_w', 0.0, 1.0),
            'rbfnn_k': hp.quniform('batch_update', 1, 10, 1),
            'rbfnn_epochs': hp.choice('rbfnn_epochs', [10, 50, 100, 150]),
            'rbfnn_lr': hp.loguniform('rbfnn_lr', np.log(0.01),np.log(0.2)),
        }
        return space

class Parameters(Parser):
    
    def __init__(self, **kwargs):
        kwargs['add_help'] = False
        super().__init__(**kwargs)
        
        self.title = RBFNN.__name__
        self.prog = "rbfnn"
        self.description = "Radial Basis Functions Neural Network model"
        
        self.add_argument("--name", help="Name of the model. Default value = %(default)s", default=self.title)
        self.add_argument("--gu", type=int, help="The generations' number for evaluating the entire population in the original models (generation strategy). Type of data: integer. Required argument. Default value = %(default)s",default=5)
        self.add_argument("--train-rep", type=str, help="Representation type used for the surrogate model train set. 'all' = A Vector with all values, 'allnorm' = A normalized vector with all values, 'numcuts' = Vector with only number of cuts, 'stats' = Vector with stats values, 'cutdits' = Vector with cut distributions. Type of data: string. Default value = %(default)s",default="all")
        self.add_argument("--dist-metric", type=str, choices=RBFNN.distances_allowed,help="List of distance metric: {dists}. Type of data: string. Default value = %(default)s".format(dists=RBFNN.distances_allowed),default=RBFNN.distances_allowed[0])
        self.add_argument("--dtw-sakoechiba-w", type=float, help="Window size rate for sakoechiba dtw constraint. Type of data: double. Default value = %(default)s", default=0.1)        
        self.add_argument("--rbfnn-func", choices=['multiquadratic', 'inverse_multiquadratic','absolute','gaussian','linear','cubic','thin_plate'], type=str, help="Possible Radial Basis Functions to use in the surrogate model: 'multiquadratic', 'inverse_multiquadratic', 'absolute', 'gaussian', 'linear', 'cubic', 'thin_plate'. Type of data: string. Default value = %(default)s", default='gaussian')
        self.add_argument("--rbfnn-epsilon", type=float, help="RBF epsilon: 0: compute from std of data. Type of data: float. Default value = %(default)s", default=0)
        self.add_argument("--rbfnn-k", type=int, help="Number of cluster for RBF Neural Network. Type of data: int.Default value = %(default)s", default=2)
        self.add_argument("--rbfnn-epochs", type=int, help="Number of epochs to train the RBF Neural Network. Type of data: int. Default value = %(default)s", default=100)
        self.add_argument("--rbfnn-lr", type=float, help="Learning rate to train the RBF Neural Network. Type of data: float. Default value = %(default)s", default=0.01)

    def handle(self, arguments):
        print(arguments)

class RBFNN(ModelBase):
    """Implementation of a Radial Basis Function Network"""
    
    distances_allowed = conf.DISTANCES["RBFNN"]
    
    def __init__(self, id_model, options=None):
        super().__init__(id_model=id_model, options=options)
        self.options = options
        self.w = np.random.randn(self.options.model[self.id_model].rbfnn_k)
        self.b = np.random.randn(1)
        self.kmeans = TimeSeriesKMeans(n_clusters=self.options.model[self.id_model].rbfnn_k, metric=self.options.model[self.id_model].dist_metric)
        self.centers = []
        
    def get_name(self):
        return self.class_name+self.options.model[self.id_model].dist_metric.upper()+"("+str(self.options.model[self.id_model].gu)+"_"+str(int(self.options.model[self.id_model].dtw_sakoechiba_w*100))+"_"+str(self.options.model[self.id_model].rbfnn_k)+"_"+str(self.options.model[self.id_model].rbfnn_epochs)+"_"+str(int(self.options.model[self.id_model].rbfnn_lr*100))+")"
    
    def gaussian(self, x, c, gamma=1.0):
        distance = self.distance_measure.distance(dataset1=x, dataset2=c, inv=True)
        kernel_value = np.exp(-gamma * distance ** 2)
        return kernel_value
        
    def rbf(self, x, c, gamma=1.0):
        # Compute DTW distance
        return eval("self."+self.options.model[self.id_model].rbfnn_func)(x,c,gamma)
        
    def train(self):
        
        xtrain, classes = self.training_set.to_train_set(self.id_model)
        self.ytrain = classes[:,self.id_model]
        self.training_number += 1
        self.is_trained = True
        
        self.xtrain = to_time_series_dataset(xtrain)
        self.kmeans.fit(self.xtrain)
        self.centers = self.kmeans.cluster_centers_
        
        dMax = np.max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
        self.stds = np.repeat(dMax / np.sqrt(2*self.options.model[self.id_model].rbfnn_k), self.options.model[self.id_model].rbfnn_k)
        # training
        for _ in range(self.options.model[self.id_model].rbfnn_epochs):
            for i in range(self.xtrain.shape[0]):
                # forward pass
                a = np.array([self.rbf(self.xtrain[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b
                # backward pass
                error = -(self.ytrain[i] - F).flatten()
                # online update
                self.w = self.w - self.options.model[self.id_model].rbfnn_lr * a.flatten() * error
                self.b = self.b - self.options.model[self.id_model].rbfnn_lr * error 
                
    def predict(self, x):
        y_pred = []
        X = np.array(x)
        self.xtest = x.copy()
        self.matrix_distances = self.distance_measure.distance(self.xtest, self.xtrain)
        X = to_time_series_dataset(X)
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F.item())
        return np.array(y_pred)
    
    def classifier(self, X_test, y_test):
        y_pred = self.predict_set(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse
    
    def copy(self):
        rbfnet = RBFNN(id_model=self.id_model, options=self.options)
        rbfnet.w = self.w.copy()
        rbfnet.b = self.b if self.b else None
        rbfnet.centers = self.centers if hasattr(self, 'centers') else None
        rbfnet.stds = self.stds if hasattr(self, 'stds') else None
        return rbfnet
    
    def export_matlab(self):
        data = {}
        data["id_model"] = self.id_model
        data["name"] = self.class_name
        data["training_set"] = self.training_set.export_matlab(isAccumulated=False)
        data["training_number"] = self.training_number
        data["archive"] = self.archive.export_matlab()
        return data
