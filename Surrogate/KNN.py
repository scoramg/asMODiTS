import numpy as np
from scipy.stats import mode
from scipy import mean
from Surrogate.ModelBase import ModelBase
from RegressionMeasures.regression_measures import RegressionMeasures
from Utils.parser import Parser
import Settings as conf
from argparse import Namespace
from hyperopt import hp

class Optimizer:
    def __init__(self):
        pass
    
    def to_namespace(self, params):
        args = {
            "name": 'KNN',
            "knn_k": params["knn_k"],
            "gu": 5,
            "ue": "archive",
            "dist_metric": "dtw",
            "dtw_sakoechiba_w": params["knn_dtw_sakoechiba_w"],
            "train_rep": params["knn_train_rep"],
        }
        
        return Namespace(**args)
    
    def space(self, label):
        space = {
            label: 'knn', 
            'knn_k': hp.choice('knn_k', [1, 3, 5, 7, 9]),
            'knn_dtw_sakoechiba_w': hp.uniform('knn_dtw_sakoechiba_w', 0.0, 1.0),
            'knn_train_rep': hp.choice('knn_train_rep', ['all', 'allnorm'])
        }
        return space

class Parameters(Parser):
    def __init__(self, **kwargs):
        kwargs['add_help'] = False
        super().__init__(**kwargs)
        
        self.title = "KNN"
        self.prog = "knn"
        self.description = "KNN Regressor model"
        
        self.add_argument("--name", help="Name of the model. Default value = %(default)s", default=self.title)
        self.add_argument("--knn-k", type=int, help="Number of neighbors for the kNN algorithm and the number of clusters for RBF. Type of data: integer. Default value = %(default)s", default=1, required=True)
        self.add_argument("--ue", type=str, help="Update strategy. 'front': Updating using first Pareto front, 'random': Updating using random schemes, 'archive': Updating using an archive. Type of data: integer. Default value = %(default)s",default='front')
        self.add_argument("--gu", type=int, help="The generations' number for evaluating the entire population in the original models (generation strategy). Type of data: integer. Required argument. Default value = %(default)s",default=5)
        self.add_argument("--train-rep", type=str, help="Representation type used for the surrogate model train set. 'all' = A Vector with all values, 'allnorm' = A normalized vector with all values, 'numcuts' = Vector with only number of cuts, 'stats' = Vector with stats values, 'cutdits' = Vector with cut distributions. Type of data: string. Default value = %(default)s",default="all")
        self.add_argument("--dist-metric", type=str, choices=KNN.distances_allowed, help="List of distance metric: {dists}. Type of data: string. Default value = %(default)s".format(dists=KNN.distances_allowed),default=KNN.distances_allowed[0])
        #self.add_argument("--dtw-dist", type=str, choices=['square', 'absolute','precomputed'], help="Distance used by DTW for chech similarity among subsequences: 'square' (default), 'absolute', 'precomputed' or callable. Type of data: string. Default value = %(default)s",default='square')
        self.add_argument("--dtw-sakoechiba-w", type=float, help="Window size rate for sakoechiba dtw constraint. Type of data: double. Default value = %(default)s", default=0.1)        
    
    def handle(self, arguments):
        print(arguments)

# Modified from https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping
class KNN(ModelBase):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays

    Arguments
    ---------
    n_neighbors : int, optional (default = 1)
        Number of neighbors to use by default for KNN
    """
    
    distances_allowed = conf.DISTANCES["KNN"]

    def __init__(self, id_model, options=None):
        super().__init__(id_model=id_model, options=options)

    def get_name(self):
        #print("KNN.get_name.self.options.model[self.id_model].dist_metric:",self.options.model[self.id_model].dist_metric, "KNN.get_name.self.options.model[self.id_model].gu:",self.options.model[self.id_model].gu)
        return str(self.options.model[self.id_model].knn_k)+"NN"+self.options.model[self.id_model].dist_metric.upper()+"("+self.options.model[self.id_model].ue.upper()[:2]+"_"+str(self.options.model[self.id_model].gu)+"_"+str(int(self.options.model[self.id_model].dtw_sakoechiba_w*100))+")"#+self.options.model[self.id_model].ue[0].upper()
    
    #def fit(self, training_set):
        """Fit the model using x as training data and y as class labels

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer

        y : array of shape [n_samples]
            Training labels for input into KNN classifier
        """
        
    def train(self):        
        self.xtrain, classes = self.training_set.to_train_set(self.id_model)
        #print("knn.classes:", classes)
        self.ytrain = classes[:,self.id_model]
        self.training_number += 1
        self.is_trained = True

    def predict(self, x):
        """Predict the class labels or probability estimates for
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified

        Returns
        -------
          The predicted class labels
        """        
        self.xtest = x.copy()
        np.random.seed(0)
        self.matrix_distances = self.distance_measure.distance(self.xtest, self.xtrain)
        
        # Identify the k nearest neighbors
        knn_idx = self.matrix_distances.argsort()[:, :self.options.model[self.id_model].knn_k]
        
        # Identify k nearest labels
        try:
            knn_labels = self.ytrain[knn_idx]
        except IndexError:
            print("KNN.predict.knn_idx: ", knn_idx, "KNN.predict.len(self.ytrain): ",len(self.ytrain), "self.matrix_distances.shape: ",self.matrix_distances.shape)
            exit(1)
        
        if self.options.task == 'regression': #Regression
            labels = mean(knn_labels, axis=1)
        else: #Classification
            class_data = mode(knn_labels, axis=1)
            labels = class_data[0]
        return labels.ravel()

    def export_matlab(self):
        data = {}
        data["id_model"] = self.id_model
        data["name"] = self.class_name
        data["k"] = self.options.model[self.id_model].knn_k
        data["training_set"] = self.training_set.export_matlab(isAccumulated=False)
        data["training_number"] = self.training_number
        data["archive"] = self.archive.export_matlab()
        #self.archive = Archive(options=options)
        return data
    
    def copy(self):
        knn = KNN(id_model=self.id_model, options=self.options)
        #knn.fit(training_set=self.training_set)
        #knn.train()
        return knn
    
    
    '''def classifier(self, x, y):
        """
        Predict the class labels or probability estimates for
        the provided data and then evaluates the accuracy score.

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified

          y : array of shape [n_samples]
              Array containing the labels of the testing dataset to be classified

        Returns
        -------
          1 floating point value representing the accuracy of the classifier
        """
        # Predict the labels and the probabilities
        #pred_labels, pred_probas = self.predict(x)
        pred_labels = self.predict(x)

        # Ensure labels are integers
        if self.options.task == 'regression': #Regression
            y = y.astype('float32')
            pred_labels = pred_labels.astype('float32')
            accuracy = mean_squared_error(y, pred_labels)
        else: #Classification
            y = y.astype('int32')
            pred_labels = pred_labels.astype('int32')
            accuracy = accuracy_score(y, pred_labels)

        # Compute accuracy measure
        
        return accuracy
    '''
    
    
