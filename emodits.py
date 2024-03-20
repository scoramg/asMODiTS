#import argparse
import random, pickle, platform
import itertools
import numpy as np
import os, time
import eMODiTS.Population as pop
from Surrogate.ModelCollection import ModelCollection
from scipy.io import savemat
from Datasets.dataset import Dataset
from EvolutionaryMethods.nsga2 import NSGA2
from EvolutionaryMethods.pareto_front import ParetoFront as af
from Utils.utils import delete_files_pattern, find_file#, find_last_file

import multiprocessing as mp

from Surrogate.KNN import Parameters as KNNParams
from Surrogate.RBF import Parameters as RBFParams
from Surrogate.SVR import Parameters as SVRParams
from Surrogate.RBFNN import Parameters as RBFNNParams
from Utils.parser import asMODiTSParameters
from Functions.fitness_functions import FitnessFunction
from RegressionMeasures.regression_measures import RegressionMeasures

from Surrogate.KNN import Optimizer as KNNOptim
from Surrogate.SVR import Optimizer as SVROptim
from Surrogate.RBFNN import Optimizer as RBFNNOptim
from argparse import Namespace
from hyperopt import hp

__slash__ = '/'
if platform.system() == 'Windows':
    __slash__ = '\\'
#from scipy.io import loadmat

#Se usó pyts, tslearn

""" def get_first_front(ds, options, population):
    nsga = NSGA2(_ds=ds, _options=options)
    nsga.set_population(population.copy())
    nsga.FastNonDominatedSort()
    return nsga.fronts[0] """
    
class asMODiTSOptimizer:
    def __init__(self):
        self.knn_opt = KNNOptim()
        self.svr_opt = SVROptim()
        self.rbfnn_opt = RBFNNOptim()
        
    def get_options(self, params, label):
        if params[label][label] == 'knn':
            return self.knn_opt.to_namespace(params[label])
        if params[label][label] == 'svr':
            return self.svr_opt.to_namespace(params[label])
        if params[label][label] == 'rbfnn':
            return self.rbfnn_opt.to_namespace(params[label])
    
    def to_namespace(self, params):
        model1_opt = self.get_options(params,"model1")
        model2_opt = self.get_options(params,"model2")
        model3_opt = self.get_options(params,"model3")
        args = {
            "e": 1,
            "g": 100,
            "ps": 50,
            "pm": 0.2,
            "pc": 0.8,
            "ff": 0,
            "ids": [38],
            "iu": params["iu"], 
            "train_size_factor": 2,
            "task": "regression",
            "evaluation_measure": "RMSE",
            "train_rep": "allnorm",
            "error_t": 0.1,
            "batch_update": params["batch_update"],
            "model":[model1_opt, model2_opt, model3_opt]
        }
        
        return Namespace(**args)
    
    def get_space(self):
        space = {
            'model1': hp.choice('model1', [
                self.knn_opt.space('model1'),
                self.svr_opt.space('model1'),
                self.rbfnn_opt.space('model1')
            ]),
            'model2': hp.choice('model2', [
                self.knn_opt.space('model2'),
                self.svr_opt.space('model2'),
                self.rbfnn_opt.space('model2')
            ]),
            'model3': hp.choice('model3', [
                self.knn_opt.space('model3'),
                self.svr_opt.space('model3'),
                self.rbfnn_opt.space('model3')
            ]),
            'batch_update': hp.quniform('batch_update', 1, 20, 1),
            'iu': hp.uniform('reg_alpha', 0.0, 1.0),
        }
        return space


class asMODiTS:
    def __init__(self, options=None):
        self.options = options
        self.no_evaluations = 0
        self.no_training = 0
        self.time = 0
        self.train_size = self.options.ps * self.options.train_size_factor
        self.predictions_measures = {}
        self.surrogate_models = ModelCollection(options=self.options)
        self.surrogate_models.create()
        
        self.prediction_keys = []
        self.prediction_keys.append(['Generation'+str(i) for i in range(0,self.options.g)])
        self.prediction_keys.append(['Model'+str(i) for i in range(0,3)])
        self.prediction_keys.append(list(RegressionMeasures.init_measures_values().keys()))
        
        self.predictions_measures = {self.prediction_keys[1][i]: RegressionMeasures.init_measures_values() for i in range(len(self.prediction_keys[1]))}
        
        self.AccumulatedFront = af(options=self.options)
        
    def create_file_structure(self, dataset_name):
        export_filename = "e"+str(self.options.e)+"p"+str(self.options.ps)+"g"+str(self.options.g)+"_sMODiTS"+__slash__+"BU"+str(self.options.batch_update)+"_"+self.surrogate_models.name  
        results_dir = os.path.dirname(os.path.realpath(__file__))+__slash__+"Results"+__slash__+export_filename+__slash__+"MODiTS"+__slash__+dataset_name
        foldername = {
            "main":results_dir+__slash__,
            #"profiler":results_dir+"/profiler/",
            #"checkpoints":results_dir+"/checkpoints/",
            "trainings":results_dir+__slash__+"trainings"+__slash__
        }
        if self.options.checkpoints:
            foldername["checkpoints"] = results_dir+__slash__+"checkpoints"+__slash__
            
        if self.options.profilers:
            foldername["profiler"] = results_dir+__slash__+"profiler"+__slash__
            for i in range(0,self.options.e):
                foldername["profiler_e"+str(i+1)] = results_dir+__slash__+"profiler"+__slash__+"Execution"+str(i+1)+__slash__
        
        for _, values in foldername.items():
            if not os.path.isdir(values):
                os.makedirs(values)
        return foldername
    
    def restore(self, results_dir):
        execution, file = find_file(results_dir, include="checkpoint_e", exclude="_g")
        if execution > 0:
            try:
                checkpoint = pickle.load(open(results_dir+"/"+file, "rb" ))
                #self.surrogate_models.restore(ds, checkpoint["surrogate_models"])
                self.AccumulatedFront.restore(checkpoint["AccumulatedFront"])
                self.no_training = checkpoint["no_training"]
                self.no_evaluations = checkpoint["no_evaluations"]
                self.predictions_measures = checkpoint["predictions_measures"]
                self.time = checkpoint["time"]
            except pickle.UnpicklingError as e:
                print("eMODiTS.Error: Corrupt checkpoint.")                               
        return execution
    
    def create_checkpoint(self, e, dirs):
        delete_files_pattern(dirs["checkpoints"],"checkpoint_e{e}.pkl".format(e=e-1))
        cp = dict(AccumulatedFront=self.AccumulatedFront.get_fronts_checkpoint(), no_training=self.no_training, no_evaluations=self.no_evaluations, predictions_measures=self.predictions_measures, time=self.time)
        pickle.dump(cp, open(dirs["checkpoints"]+"checkpoint_e"+str(e)+".pkl", "wb" ), protocol=pickle.HIGHEST_PROTOCOL)
    
    def execute(self, iDS):
        ds = Dataset(iDS, '_TRAIN', False)
        dirs = self.create_file_structure(ds.name)
        self.AccumulatedFront.ds=ds
       
        if self.options.checkpoints:
            e_ini = self.restore(dirs["checkpoints"]) + 1
        else:
            e_ini = 0
        
        _in = False
        #print("emodits.execute.e_ini",e_ini)
        for e in range(e_ini,self.options.e):
            _in = True
            start_time = time.time()
            evals_by_exec = 0
            initial_population = pop.Population(_ds=ds, options=self.options)
            initial_training_set = pop.Population(_ds=ds, pop_size=self.train_size, options=self.options)
            evals_by_exec += initial_training_set.evaluate()
            self.surrogate_models.train(initial_training_set)
            randomindexes = random.sample(range(0, self.train_size), self.options.ps)
            for i in randomindexes:
                initial_population.add_individual(initial_training_set.individuals[i])
            nsga2 = NSGA2(_ds=ds, _surrogate_models=self.surrogate_models, _options=self.options)
            nsga2.execute(e+1, dirs=dirs, population=initial_population)
            evals_by_exec += nsga2.no_evaluations
            #for f in nsga2.fronts[0]:
            #    self.AccumulatedFront.add_individual(f)
            self.AccumulatedFront.addIndividualsForFront(nsga2.fronts[0])
            self.no_training += self.surrogate_models.training_number
            self.no_evaluations += evals_by_exec
            #exec_front = pop.Population(ds, options=self.options)
            #exec_front.addIndividuals(nsga2.fronts[0])
            #mat_exec = exec_front.export_matlab(isAccumulated=False)
            mat_exec = nsga2.fronts[0].export_matlab(isAccumulated=False)
            mat_exec["evaluations"] = evals_by_exec
            mat_exec["predictions"] = nsga2.surrogate_errors
            mat_exec["no_training"] = self.surrogate_models.training_number
            for i,j,k in list(itertools.product(np.arange(0,len(self.prediction_keys[0])),np.arange(0,len(self.prediction_keys[1])),np.arange(0,len(self.prediction_keys[2])))):
                try:
                    self.predictions_measures[self.prediction_keys[1][j]][self.prediction_keys[2][k]] += nsga2.surrogate_errors[self.prediction_keys[0][i]][self.prediction_keys[1][j]][self.prediction_keys[2][k]]
                except KeyError as e:                    
                    print("eMODiTS.Error on Dataset {dataset}: KeyError {err}".format(dataset=ds.name, err=e))
                    print("eMODiTS.nsga2.surrogate_errors.keys:", nsga2.surrogate_errors.keys())
                    print("eMODiTS.predictions_measures.keys:", self.predictions_measures.keys())
                    raise
            end_time = int(time.time() - start_time) * 1000
            self.time += end_time
            mat_exec["time"] = end_time #Milisegundos                
            savemat(dirs["main"]+ds.name+"_exec"+str(e)+"_MODiTS.mat", mat_exec, long_field_names=True)
            trainings = self.surrogate_models.export_matlab()
            savemat(dirs["trainings"]+ds.name+"_train"+str(e)+".mat", trainings, long_field_names=True)
            if self.options.checkpoints:
                self.create_checkpoint(e, dirs) #Checkpoint for the execution
        if _in:
            for i,j in list(itertools.product(np.arange(0,len(self.prediction_keys[1])),np.arange(0,len(self.prediction_keys[2])))):
                self.predictions_measures[self.prediction_keys[1][i]][self.prediction_keys[2][j]] = self.predictions_measures[self.prediction_keys[1][i]][self.prediction_keys[2][j]] / self.options.e
            #first_front = self.AccumulatedFront.get_first_front()
            nsga2_accum = NSGA2(_ds=ds, _surrogate_models=self.surrogate_models, _options=self.options)
            #print("emodits.execute.self.AccumulatedFront.size", self.AccumulatedFront.size)
            first_front = nsga2_accum.get_first_front(population=self.AccumulatedFront.points, is_already_sorted=False)
            mat = first_front.export_matlab()
            mat["time"] = self.time #Milisegundos
            mat["evaluations"] = self.no_evaluations
            mat["no_training"] = self.no_training
            mat["predictions_measures_avg"] = self.predictions_measures
            mat["predictions_measures"] = first_front.points.prediction_power(self.surrogate_models)
            savemat(dirs["main"]+ds.name+"_MODiTS.mat", mat, long_field_names=True)
            trainings = self.surrogate_models.export_matlab()
            savemat(dirs["trainings"]+ds.name+"_MODiTS_trainings.mat", trainings, long_field_names=True)
        #print("smodits.execute._in:",_in)
        # pr.disable()
        # sortby = SortKey.CUMULATIVE
        # ExportProfileToFile = dirs["profiler"] + "ProfilerResults_"+self.surrogate_models.name+"_"+time.strftime("%Y-%m-%d-%H-%M-%S")+".txt"
        
        # with open(ExportProfileToFile, 'w') as stream:
        #     stats = pstats.Stats(pr, stream=stream).sort_stats(sortby)
        #     stats.print_stats()
    
if __name__ == "__main__":
    
    commands = [
        KNNParams,
        RBFParams,
        SVRParams,
        RBFNNParams
    ]
    parser = asMODiTSParameters(commands=commands)
    options=parser.parse_args()
    
    #Falta comparar los argumentos para hacer que no se repita o empiece desde un valor si cambia un parametro que no esta en el nombre del archivo
    #print("emodits.main.options:", options)
    ids = options.ids
    
    assert len(options.model) == FitnessFunction(idconfiguration=options.ff).no_functions, "The surrogate models' number ({}) is less than or greater than the objective functions' number({}).".format(len(options.model),FitnessFunction(idconfiguration=options.ff).no_functions)
    method = asMODiTS(options=options)
    
    workers = mp.cpu_count()
    chunksz = 1
    p = mp.Pool(workers)
    p.map(method.execute, [id for id in ids], chunksize=chunksz)

