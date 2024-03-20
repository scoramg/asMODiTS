import argparse
import numpy as np
#import shelve
#from scipy.spatial.distance import squareform, cdist, pdist
from DistanceMeasures.dtw import DTW
from DistanceMeasures.tga import TGA
from DistanceMeasures.gak import GAK
from Surrogate.ModelBase import ModelBase
#from sklearn.linear_model import LinearRegression
#from numba import jit, prange
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
        self.add_argument("--ue", type=str, help="Update strategy. 'front': Updating using first Pareto front, 'random': Updating using random schemes, 'archive': Updating using an archive. Type of data: integer. Default value = %(default)s",default='front')
        self.add_argument("--gu", type=int, help="The generations' number for evaluating the entire population in the original models (generation strategy). Type of data: integer. Required argument. Default value = %(default)s",default=5)
        self.add_argument("--train-rep", type=str, help="Representation type used for the surrogate model train set. 'all' = A Vector with all values, 'allnorm' = A normalized vector with all values, 'numcuts' = Vector with only number of cuts, 'stats' = Vector with stats values, 'cutdits' = Vector with cut distributions. Type of data: string. Default value = %(default)s",default="all")
        self.add_argument("--dist-metric", type=str, choices=RBF.distances_allowed,help="List of distance metric: {dists}. Type of data: string. Default value = %(default)s".format(dists=RBF.distances_allowed),default=RBF.distances_allowed[0])
        #self.add_argument("--dtw-dist", type=str, choices=['square', 'absolute','precomputed'], help="Distance used by DTW for chech similarity among subsequences: 'square' (default), 'absolute', 'precomputed' or callable. Type of data: string. Default value = %(default)s",default='square')
        self.add_argument("--dtw-sakoechiba-w", type=float, help="Window size parameter for sakoechiba dtw constraint. Type of data: double. Default value = %(default)s", default=0.1)        
        self.add_argument("--rbf-func", choices=['multiquadratic', 'inverse_multiquadratic','absolute','gaussian','linear','cubic','thin_plate'], type=str, help="Possible Radial Basis Functions to use in the surrogate model: 'multiquadratic', 'inverse_multiquadratic', 'absolute', 'gaussian', 'linear', 'cubic', 'thin_plate'. Type of data: string. Default value = %(default)s", default='gaussian')
        self.add_argument("--rbf-epsilon", type=float, help="RBF epsilon: 0: compute from std of data. Type of data: float. Default value = %(default)s", default=0)
    
    def handle(self, arguments):
        print(arguments)

# Class to create or use an RBF surrogate model
class RBF(ModelBase):
    
    distances_allowed = conf.DISTANCES["RBF"]
    
    def __init__(self, id_model, options=None):
        super().__init__(id_model=id_model, options=options)
        self.options = options
        self.weights = None
        self.epsilon = None
        #self.dist_threshold = self.options.dist_t
        
    def get_name(self):
        return self.class_name+self.options.model[self.id_model].dist_metric.upper()+"("+self.options.model[self.id_model].ue.upper()[:2]+"_"+str(self.options.model[self.id_model].gu)+"_"+str(int(self.options.model[self.id_model].dtw_sakoechiba_w*100))+"_"+self.options.model[self.id_model].rbf_func[:2].upper()[:2]+"_"+str(self.options.model[self.id_model].rbf_epsilon)+"]"
    
    # Collection of possible Radial Basis Functions to use in the surrogate model:
    # Multiquadratic
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
        """ print("epsilon: ",self.epsilon)
        print("r: ",r)
        print("self.epsilon * r: ",self.epsilon * r)
        print("-(self.epsilon * r): ", -(self.epsilon * r))
        print("np.exp(-(self.epsilon * r)): ", np.exp(-(self.epsilon * r))) """
        #return np.exp(-(self.epsilon * r) ** 2)
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
            #print("np.std(self.ytrain): ", np.std(self.ytrain))
            #self.epsilon = 1/(2*(np.std(self.ytrain)**2))
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
        # Dictionary object to store possible RBFs and associated functions to evaluate them
        # Can add as needed when a new function is added to the collection above
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
        #print("RBF.train.r:",r)
        self._compute_epsilon(data=self.ytrain)
        #print("RBF.train.epsilon:",self.epsilon)
        N = self._compute_N(r)  # Compute the basis function matrix of the specified type
        #print("RBF.train.N:",N)
        try:
            self.weights = np.linalg.solve(N, self.ytrain)  # Solve for the weights vector
            #print("weights: ",self.weights, len(self.weights), N.shape)
            #print("N: ",np.array(N),",x: ",self.x,"y: ", np.array(self.y))
        except np.linalg.LinAlgError as err:
            print(r)
            unq, count = np.unique(N, axis=0, return_counts=True)
            print("unq:", unq)
            print("count:", count)
            print("Error hay instancias repetidas en el conjunto de entrenamiento.")
            exit(1)
            #print("*****Error N: ",np.array(N),",x: ",self.x,"y: ", np.array(self.y))
          
    # Function to use a previously trained RBF surrogate for predicting at new x locations
    def predict(self, x):
        self.matrix_distances = self._compute_r(x, self.xtrain)  # Compute the euclidean distance matrix
            #print("self.matrix_distances (2):", self.matrix_distances)
        #self._compute_epsilon(self.matrix_distances)
        N = self._compute_N(self.matrix_distances)  # Compute the basis function matrix of the trained type
        #print("RBF.Predict.N:",N)
        #print("RBF.Predict.weights:",self.weights)
        predictions = np.matmul(N, self.weights)
        #print("predictions:", predictions)
        return predictions  # Use the surrogate to predict new y values
    
    """ def fit(self, x, y, r=None):
        self.x = x.copy()
        self.y = y.copy()
        self._train(r) """
    
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
    
    
    
    
    # Initialization for the RBF class
    # Defaults are specified for the options, required to pass in whether you are training or predicting
    ''' Tengo que cambiar la inicialización '''
#     def __init__(self, type, x_file='x_train.dat', y_file='y_train.dat', model_db='model.db', rbf_func='gaussian'):
#         self.x_data = np.loadtxt(x_file, skiprows=1, delimiter=",")  # Read the input locations file
#         self.x_data = self.x_data.reshape(self.x_data.shape[0], -1)  # Reshape into 2D matrix (avoids array issues)
#         self.rbf_func = rbf_func  # Read user specified options (or the defaults)
#         self.model_db = model_db  # Read user specified options (or the defaults)

#         # Check for training or prediction
#         if type == 'train':
#             self.y_data = np.loadtxt(y_file, skiprows=1, delimiter=",")  # Read output data file
#             self.y_data = self.y_data.reshape(self.y_data.shape[0], -1)  # Reshape into 2D matrix (avoids array issues)

#             # Compute epsilon based on the standard deviation of the output values for the RBFs that use it
#             self.epsilon = np.std(self.y_data)

#             self._train()  # Run the model training function

#             # Store model parameters in a Python shelve database
#             model_data = shelve.open(model_db)
#             model_data['rbf_func'] = self.rbf_func
#             model_data['epsilon'] = self.epsilon
#             model_data['x_train'] = self.x_data
#             model_data['weights'] = self.weights
#             print('\nSurrogate Data:')
#             print('RBF Function: ', model_data['rbf_func'])
#             print('Epsilon: ', model_data['epsilon'])
#             print('Radial Basis Function Weights: ', '\n', model_data['weights'])
#             print('\n', 'Trained surrogate stored in: ', self.model_db)
#             model_data.close()

#         else:
#             # Read previously stored model data from the database
#             model_data = shelve.open(model_db)
#             self.rbf_func = model_data['rbf_func']
#             self.epsilon = model_data['epsilon']
#             self.x_train = model_data['x_train']
#             self.weights = model_data['weights']

#             model_data.close()

#             print('\nUsing', self.model_db, 'to predict values...')
#             self._predict()  # Run the model prediction function

#             # Quick loop to add a header that matches the input file format
#             y_head = []
#             for i in range(self.y_pred.shape[1]):
#                 y_head.append('y' + str(i))

#             # Convert header list of strings to a single string with commas and write out the predictions to a file
#             header = ','.join(y_head)
#             np.savetxt('y_pred.dat', self.y_pred, delimiter=',', fmt="%.6f", header=header, comments='')
#             print('Predicted values stored in \"y_pred.dat\"')


# # Code to run when called from the command line (usual behavior)
#if __name__ == "__main__":
#    rbf = RBF(no_class=1)
#    print(rbf.get_name())
    
#     # Parse the command line input options to "opt" variable when using on the command line
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-t', '--type', dest='type', choices=['train', 'predict'], required=True,
#                         help="""Specify whether the tool is to be used for training with \"train\" or
#                         making predictions with a stored model with \"predict\".""")
#     parser.add_argument('-x', dest='x_file', default='x_train.dat',
#                         help="""Input file of x locations. Default is \"x_train.dat\".""")
#     parser.add_argument('-y', dest='y_file', default='y_train.dat',
#                         help="""Output file for surrogate training. Default is \"y_train.dat\".""")
#     parser.add_argument('-m', '--model', dest='model_file', default='model.db',
#                         help="""File to save the model output or use a previously trained model file.
#                         Default is \"model.db\".""")
#     parser.add_argument('-r', '--rbf', dest='rbf', default='gaussian',
#                         help="""Specified RBF to use when training the surrogate. Default is \"gaussian\".""")
#     opts = parser.parse_args()

#     # Create and run the RBF class object
#     surrogate = RBF(opts.type, opts.x_file, opts.y_file, opts.model_file, opts.rbf)
