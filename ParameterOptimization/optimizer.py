import sys, os, csv, ast, time
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, hp, tpe, fmin, Trials
from argparse import Namespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from Datasets.dataset import Dataset
from Surrogate.KNN import KNN, Optimizer as KNNOptim
from Surrogate.SVR import SVR, Optimizer as SVROptim
from Surrogate.RBFNN import RBFNN, Optimizer as RBFNNOptim
from Functions.fitness_functions import FitnessFunction as FF
from RegressionMeasures.regression_measures import RegressionMeasures
from eMODiTS.Population import Population
from Utils.parser import Parser


__exportdir__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/HyperOptimization/"+time.strftime("%Y-%m-%d")+"/"
__version__ = 0.01
__timeformat__ = time.strftime("%H-%M-%S")

class OptimizerParameters(Parser):
    def __init__(self):
        super().__init__(description='Bayesian hyperparameter optimizer using the hyperopt package', prefix='USAGE: ', title='eMODiTS')
        self.global_arguments = ['--version', '--help']
        self.add_argument(
            "-v",
            "--version",
            action="version",
            help="show program's version number and exit",
            version=__version__,
        )
        self.add_argument("-train-size", type=int, help="Training set size.", required=False)
        self.add_argument("-test-size", type=int, help="Testing set size.", required=False)
        self.add_argument("-ff", type=int, help="Fitness function configuration. 0: Entropy, complexity, infoloss. Type of data: integer. Default value = %(default)s",default=0)
        self.add_argument("-loss-function", type=str, help="Metric used as function loss to evaluate the trials. 'MSE' (Mean Square Error), 'R' (R Coefficient), 'R2' (R Squared), 'RMSE' (Root Mean Square Error), 'MD' (Modified Index of acceptance), 'MAPE' (Mean Absolute Percentage Error). Type of data: string. Default value = %(default)s", default='RMSE')
        self.add_argument("-max-evals", type=int, help="Maximum number of evaluations to find the best parameter configurations. Default = %(default)s",default=0)
        self.add_argument("-method", choices=['knn', 'svr','rbfnn', 'all'], type=str, help="Surrogate model to optimize. Default = %(default)s",default="all")
        self.add_argument("-groupedds", help="Use one training and testing set containing a set of datasets. ", action='store_true')
        self.add_argument("-no-groupedds", help="Use training and testing set separately. ", dest='groupedds', action='store_false')
        self.set_defaults(groupedds=True)

class BayessianOptimizer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.knn_opt = KNNOptim()
        self.svr_opt = SVROptim()
        self.rbfnn_opt = RBFNNOptim()
        self.iteration = 0
        self.regression_measures = RegressionMeasures()
        
        self.out_file = "{dir}{measure}/{meth} ({hour})/hp_trials_{func}_{measure}_{eval}evals_{train_size}X{test_size}_{meth}.csv".format(dir=__exportdir__,func=self._functions.functions_name[self.id_function], measure = self.options.loss_function, eval=self.options.max_evals, train_size=self.options.train_size, test_size=self.options.test_size, meth=self.options.method, hour=__timeformat__)
        self._create_output_file()
        
    def _create_output_file(self):
        if not os.path.isdir(os.path.dirname(self.out_file)):
            os.makedirs(os.path.dirname(self.out_file))
        of_connection = open(self.out_file, 'w')
        writer = csv.writer(of_connection)

        # Write the headers to the file
        writer.writerow(['loss', 'params', 'iteration', 'train_time'])
        of_connection.close()
        
    def _space(self):
        model_space = []
        if self.options.method in ['knn', 'all']:
            model_space.append(self.knn_opt.space('model'))
        if self.options.method in ['svr', 'all']:
            model_space.append(self.svr_opt.space('model'))  
        if self.options.method in ['rbfnn', 'all']:
            model_space.append(self.rbfnn_opt.space('model')) 
            
        if not self.options.groupedds:
            space = {
                'model': hp.choice('model', model_space),
                'ids': hp.choice('ids',self.ids)
            }
        else:
            space = {
                'model': hp.choice('model', model_space)
            }
             
        return space

    def _options(self, params, label):
        if params[label][label] == 'knn':
            return self.knn_opt.to_namespace(params[label])
        if params[label][label] == 'svr':
            return self.svr_opt.to_namespace(params[label])
        if params[label][label] == 'rbfnn':
            return self.rbfnn_opt.to_namespace(params[label]) 
        
    def _general_options(self):
        args = {
            "ff": self._functions.idconfiguration,
            "g": 0,
            "task": "regression"
        }
        return Namespace(**args)
        
    def execute_model(self, options,iDS=None):
        
        model_options = [None, None, None]
        model_options[self.id_function] = options
        gen_options = self._general_options()
        gen_options.__setattr__("model",model_options)
        
        if not self.options.groupedds:
            train, test = self.get_train_test_sets([iDS], gen_options)
        else:
            train, test = self.get_train_test_sets(self.ids, gen_options)
        self.model = eval(options.name)(self.id_function, options=gen_options)
        self.model.fit(training_set=train)
        self.model.train()
        test_data, test_classes = test.to_train_set(id_model=self.id_function)
        y_pred = self.model.predict(test_data)
        self.regression_measures.setObserved(observed=np.array(test_classes[:,self.id_function]))
        self.regression_measures.setPredicted(predicted=np.array(y_pred))
        self.regression_measures.compute()

    def objective(self, params):
        """Returns validation score from hyperparameters"""
        self.iteration += 1
        options = self._options(params=params,label='model')
        start = timer()
        if self.options.groupedds:
            self.execute_model(options=options)
        else:
            self.execute_model(iDS=params["ids"],options=options)
        run_time = timer() - start
        best_score = np.max(self.regression_measures.values[self.options.loss_function])
        loss = 1 - best_score
        of_connection = open(self.out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([loss, params, self.iteration, run_time])
        return {'loss': loss, 'params': params, 'iteration': self.iteration,
                'train_time': run_time, 'status': STATUS_OK}
        
    def execute(self):
        space = self._space()
        bayes_trials = Trials()
        best = fmin(fn = self.objective, space = space, algo = tpe.suggest, max_evals = self.options.max_evals, trials = bayes_trials, rstate = np.random.default_rng(50))
        return best
    
    def export_best_results(self):
        results = pd.read_csv(self.out_file)
        results.sort_values('loss', ascending = True, inplace = True)
        results.reset_index(inplace = True, drop = True)
        results.head()
        best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()
        best_file = "{dir}{measure}/{meth} ({hour})/hp_best_{func}_{measure}_{eval}_{train_size}X{test_size}_{meth}.csv".format(dir=__exportdir__,func=self._functions.functions_name[self.id_function], measure = self.options.loss_function, eval=self.options.max_evals, train_size=self.options.train_size, test_size=self.options.test_size, meth=self.options.method, hour=__timeformat__)
        headers = ['loss', 'iteration', 'train_time']
        result = [results.loc[0, 'loss'], results.loc[0, 'iteration'], results.loc[0, 'train_time']]
        for key, value in best_bayes_params['model'].items():
            headers.append(key)
            result.append(value)
        of_connection = open(best_file, 'w')
        writer = csv.writer(of_connection)
        writer.writerow(headers)
        writer.writerow(result)
    
    def get_train_test_sets(self, ids, options):
        train = Population(options=options)
        test = Population(options=options)

        trainset_ds = round(self.options.train_size/len(ids))
        testset_ds = round(self.options.test_size/len(ids))

        for idx in ids:
            ds = Dataset(idx, '_TRAIN', False)
            train_aux = Population(_ds=ds, pop_size=trainset_ds, options=options)
            train_aux.evaluate()
            train.join(train_aux)
            test_aux = Population(_ds=ds, pop_size=testset_ds, options=options)
            test_aux.evaluate()
            test.join(test_aux)
        return train, test
 
if __name__ == "__main__":
    #python ParameterOptimization/optimizer.py -train-size 50 -test-size 50 -f 0 -loss-function MD -max-evals 200 -method svr
    parser = OptimizerParameters()
    opt_params=parser.parse_args()
    functions = FF(idconfiguration=opt_params.ff)
    datasets = [38,70,65,64,47,16,18,45,56,58,17,46,57,53,74,48,20,44,7,68]
    for f in range(0,functions.no_functions):
        print(functions.functions_name[f])
        optim = BayessianOptimizer(_functions = functions, id_function=f, ids=datasets, options = opt_params)
        best = optim.execute()
        optim.export_best_results()
    
    
    
    
    