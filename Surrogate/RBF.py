import argparse
import numpy as np
from DistanceMeasures.dtw import DTW
from DistanceMeasures.tga import TGA
from DistanceMeasures.gak import GAK
from Surrogate.ModelBase import ModelBase
import argparse

from Utils.parser import Parser
import Settings as conf


'''
Python Tool for Training RBF Surrogate Models

https://github.com/evanchodora/rbf_surrogate

Evan Chodora (2019)
echodor@clemson.edu

Can be used with a variety of RBFs (see the dictionary of function names below) and can be used with both
multi-dimensional inputs and multi-dimensional outputs (and scalars for both).

Makes use of the Spatial Distance calculation functions from SciPy to compute the radial distance matrices for the
radial basis function calculations.
(https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)

Included RBFs:
 - Linear: "linear"
 - Cubic: "cubic"
 - Absolute Value: "absolute"
 - Multiquadratic: "multiquadratic"
 - Inverse Multiquadratic: "inverse_multiquadratic"
 - Gaussian: "gaussian"
 - Thin Plate: "thin_plate"

Program Usage:

rbf_surrogate.py [-h] -t {train,predict} [-x X_FILE] [-y Y_FILE] [-m MODEL_FILE] [-r RBF]

optional arguments:
  -h, --help                                    Show help message and exit.
  -t {train,predict}, --type {train,predict}    Specify whether the tool is to be used for training
                                                with "train" or making predictions with a stored model
                                                with "predict". (REQUIRED)
  -x X_FILE                                     Input file of x locations. (OPTIONAL) Default is "x_train.dat".
  -y Y_FILE                                     Output file for surrogate training. (OPTIONAL) Default is
                                                "y_train.dat".
  -m MODEL_FILE, --model MODEL_FILE             File to save the model output or use a previously
                                                trained model file. (OPTIONAL) Default is "model.db".
  -r RBF, --rbf RBF                             Specified RBF to use when training the surrogate. (OPTIONAL)
                                                Default is "gaussian".

'''

class Parameters(Parser):
    
    def __init__(self, **kwargs):
        kwargs['add_help'] = False
        super().__init__(**kwargs)
        
        self.title = "RBF"
        self.prog = "rbf"
        self.description = "Radial Basis Functions model"
        
        self.add_argument("--name", help="Name of the model. Default value = %(default)s", default=self.title)
        self.add_argument("--gu", type=int, help="The generations' number for evaluating the entire population in the original models (generation strategy). Type of data: integer. Required argument. Default value = %(default)s",default=5)
        self.add_argument("--train-rep", type=str, help="Representation type used for the surrogate model train set. 'all' = A Vector with all values, 'allnorm' = A normalized vector with all values, 'numcuts' = Vector with only number of cuts, 'stats' = Vector with stats values, 'cutdits' = Vector with cut distributions. Type of data: string. Default value = %(default)s",default="all")
        self.add_argument("--dist-metric", type=str, choices=RBF.distances_allowed,help="List of distance metric: {dists}. Type of data: string. Default value = %(default)s".format(dists=RBF.distances_allowed),default=RBF.distances_allowed[0])
        self.add_argument("--dtw-sakoechiba-w", type=float, help="Window size parameter for sakoechiba dtw constraint. Type of data: double. Default value = %(default)s", default=0.1)        
        self.add_argument("--rbf-func", choices=['multiquadratic', 'inverse_multiquadratic','absolute','gaussian','linear','cubic','thin_plate'], type=str, help="Possible Radial Basis Functions to use in the surrogate model: 'multiquadratic', 'inverse_multiquadratic', 'absolute', 'gaussian', 'linear', 'cubic', 'thin_plate'. Type of data: string. Default value = %(default)s", default='gaussian')
        self.add_argument("--rbf-epsilon", type=float, help="RBF epsilon: 0: compute from std of data. Type of data: float. Default value = %(default)s", default=0)
    
    def handle(self, arguments):
        print(arguments)

class RBF(ModelBase):
    
    distances_allowed = conf.DISTANCES["RBF"]
    
    def __init__(self, id_model, options=None):
        super().__init__(id_model=id_model, options=options)
        self.options = options
        self.weights = None
        self.epsilon = None
        
    def get_name(self):
        return self.class_name+self.options.model[self.id_model].dist_metric.upper()+"("+str(self.options.model[self.id_model].gu)+"_"+str(int(self.options.model[self.id_model].dtw_sakoechiba_w*100))+"_"+self.options.model[self.id_model].rbf_func[:2].upper()[:2]+"_"+str(self.options.model[self.id_model].rbf_epsilon)+"]"
    
    def _multiquadric(self, r):
        return np.sqrt(r ** 2 + self.epsilon ** 2)

    # Inverse Multiquadratic
    def _inverse_multiquadric(self, r):
        return 1.0 / np.sqrt(r ** 2 + self.epsilon ** 2)

    # Absolute value
    def _absolute_value(self, r):
        return np.abs(r)

    # Standard Gaussian
    def _gaussian(self, r):
        return np.exp(-(self.epsilon * r))
        
    # Linear
    def _linear(self, r):
        return r

    # Cubic
    def _cubic(self, r):
        return (r ** 3)
    
    # Cuadratic
    def _cuadratic(self, r):
        return (r ** 2)

    # Thin Plate
    def _thin_plate(self, r):
        return (r ** 2) * np.log(np.abs(r))
    
    def _compute_epsilon(self, data):
        if self.options.model[self.id_model].rbf_epsilon == 0:
            if np.any(np.isinf(data)):
                print("RBF._compute_epsilon.isinf:", data.dtype)
            print("RBF._compute_epsilon.np.std(r):",np.std(data))
            self.epsilon = 1/(2*(np.std(data)**2))
        else:
            self.epsilon = self.options.model[self.id_model].rbf_epsilon

    # Function to compute the radial distance - r = (x-c)
    def _compute_r(self, a, b=None): #b:centers
        return self.distance_measure.distance(a,b)

    def _compute_N(self, r):
        rbf_dict = {
            "multiquadratic" : self._multiquadric,
            "inverse_multiquadratic" : self._inverse_multiquadric,
            "absolute": self._absolute_value,
            "gaussian" : self._gaussian,
            "linear" : self._linear,
            "cubic" : self._cubic,
            "cuadratic" : self._cuadratic,
            "thin_plate" : self._thin_plate
        }
        return rbf_dict[self.options.model[self.id_model].rbf_func](r)

    # Function to train an RBF surrogate using the suplied data and options
    def train(self):
        self.xtrain, classes = self.training_set.to_train_set(self.id_model)
        self.ytrain = classes[:,self.id_model]
        self.training_number += 1
        self.is_trained = True
        r = self._compute_r(self.xtrain)  # Compute the euclidean distance matrix
        self._compute_epsilon(data=self.ytrain)
        N = self._compute_N(r)  # Compute the basis function matrix of the specified type
        try:
            self.weights = np.linalg.solve(N, self.ytrain)  # Solve for the weights vector
        except np.linalg.LinAlgError as err:
            print(r)
            unq, count = np.unique(N, axis=0, return_counts=True)
            print("unq:", unq)
            print("count:", count)
            print("Error hay instancias repetidas en el conjunto de entrenamiento.")
            exit(1)
            
    # Function to use a previously trained RBF surrogate for predicting at new x locations
    def predict(self, x):
        self.matrix_distances = self._compute_r(x, self.xtrain)  # Compute the euclidean distance matrix
        N = self._compute_N(self.matrix_distances)  # Compute the basis function matrix of the trained type
        predictions = np.matmul(N, self.weights)
        return predictions  # Use the surrogate to predict new y values
    
    def export_matlab(self):
        data = {}
        data["id_model"] = self.id_model
        data["name"] = self.class_name
        data["training_set"] = self.training_set.export_matlab(isAccumulated=False)
        data["training_number"] = self.training_number
        data["archive"] = self.archive.export_matlab()
        return data
        
    def copy(self):
        rbf = RBF(options=self.options, id_model=self.id_model)
        rbf.xtrain = self.xtrain 
        rbf.ytrain = self.ytrain
        rbf.epsilon = self.epsilon
        rbf.weights = self.weights
        return rbf
