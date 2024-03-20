import math
import numpy as np
from Surrogate.RBF import RBF
from Surrogate.KNN import KNN
from Surrogate.SVR import SVR
from EvolutionaryMethods.nsga2 import NSGA2
import eMODiTS.Scheme as sch
import eMODiTS.Population as pop
from RegressionMeasures.regression_measures import RegressionMeasures
from statistics import mean
import random

class Model:
    def __init__(self, options=None): 
        ''' Create a surrogate model 
            Parameters
            ds: dataset from Dataset class
            options: all params obtained from main_surrogate
        '''
                     
        """ self.repr_type=repr_type
        self.model_type = model_type
        #self.ind_upd = ind_upd
        self.update_strategy = update_strategy
        self.execution_type = execution_type """
        
        self.models = []
        self.train_set = []
        self.classes = []
        self.instances = None
        self.training_number = 0
        self.archive = []
        self.prediction_error = 0  
        self.options = options
        self.archive_size = int(round(self.options.ps/4,0))
        self.train_size = self.options.ps * self.options.train_size_factor
        
    def get_name(self):
        if self.options.model == 0:
            return str(self.options.knn_k)+"NN"+self.options.dist_metric.upper()
        if self.options.model == 1:
            return "RBF"+self.options.dist_metric.upper()+"_"+self.options.rbf_func.upper()
        if self.options.model == 2:
            return "SVR"+self.options.dist_metric.upper()
        #if self.options.model == 3:
            #return "RBFNet"+self.options.dist_metric.upper()+"_"+self.options.dtw_method.upper()
        
    def __del__(self):
        del(self.models)
        del(self.train_set)
        del(self.classes)
        del(self.instances)
        
    def delete_duplicated(self, data):
        result = []
        for sh in data:
            if not sh.is_contained_in_list(result):
                result.append(sh.copy())
        return result
        
    def add_to_archive(self, data):
        for i in range(0,len(data)):
            self.archive.append(data[i].copy())
            
        archive_aux = np.array(self.delete_duplicated(self.archive))
        
        if len(archive_aux) > self.archive_size:
            #no_ind = abs(len(archive_aux) - self.archive_size)
            #print(len(archive_aux), self.archive_size)
            randomidx = random.sample(range(0, len(archive_aux)), self.archive_size)
            #self.archive = []
            #for i in range(0,no_ind):
            #    self.archive.append(archive_aux[randomidx[i]].copy())
            self.set_to_archive(archive_aux[randomidx])
    
    def set_to_archive(self, data):
        self.archive = []
        for i in range(0,len(data)):
            self.archive.append(data[i].copy())
        
    def create(self, ds, population):
        self.instances = NSGA2(ds, options=self.options)
        self.instances.set_population(population)
        self.train()
        
    def getPredictionError(self, measures):
        prediction_error = []
        for m in range(0, len(self.models)):
            if self.options.evaluation_measure==0: #MSE
                prediction_error.append(measures[m].MSE())
            if self.options.evaluation_measure==1: #R
                prediction_error.append(measures[m].R())
            if self.options.evaluation_measure == 2: #R2
                prediction_error.append(measures[m].R2())
            if self.options.evaluation_measure == 3: #RMSE
                prediction_error.append(measures[m].RMSE())
            if self.options.evaluation_measure == 4: #Modified Index of acceptance
                prediction_error.append(measures[m].md())
            
        #print(prediction_error)
        return mean(prediction_error)
    
    def getPredictionError(self, measure):
        prediction_error = []
        if self.options.evaluation_measure==0: #MSE
            prediction_error = measure.MSE()
        if self.options.evaluation_measure==1: #R
            prediction_error = measure.R()
        if self.options.evaluation_measure == 2: #R2
            prediction_error = measure.R2()
        if self.options.evaluation_measure == 3: #RMSE
            prediction_error = measure.RMSE()
        if self.options.evaluation_measure == 4: #Modified Index of acceptance
            prediction_error = measure.md()
        
        #print(prediction_error)
        return prediction_error
    
    def evaluate(self, front=None):
        no_evaluations=0
        measures = RegressionMeasures()
        individual_evaluated = []
        if self.options.ue == 1: #actualización usando el frente de pareto
            '''Así se hizo en el primer año'''
            if front:
                for f in front:
                    sh = f.copy() #Se corrigió este error
                    if not sh.is_contained_in_list(individual_evaluated):
                        no_evaluations += sh.evaluate()  
                        individual_evaluated.append(sh.copy()) 
                        for m in range(0, len(self.models)):
                            measures.addObservedValue(sh.fitness_functions.values[m])
                            measures.addPredictedValue(f.fitness_functions.values[m])
                # Obtenemos el error
                self.prediction_error = self.getPredictionError(measure=measures)
                
        if self.options.ue == 2:
            for i in range(0, self.options.iu):
                individual = sch.Scheme(self.instances.ds, cuts={}, options=self.options)
                surr_individual = individual.copy()
                no_evaluations += individual.evaluate()
                surr_individual.evaluate(self)   
                individual_evaluated.append(individual.copy()) 
                for m in range(0, len(self.models)):
                    measures.addObservedValue(individual.fitness_functions.values[m])
                    measures.addPredictedValue(surr_individual.fitness_functions.values[m])
            # Obtenemos el error
            self.prediction_error = self.getPredictionError(measure=measures) 
            
        if self.options.ue == 3:
            ''' Cuantos esquemas se ingresaran en el train? '''
            measures = RegressionMeasures()
            for i in range(0, self.options.iu):
                #print(len(self.archive))
                if self.archive:
                    sh = self.archive[0].copy()
                    no_evaluations += sh.evaluate()
                    for m in range(0, len(self.models)):
                        measures.addObservedValue(sh.fitness_functions.values[m])
                        #print(len(self.archive[0].fitness_functions.values), m)
                        measures.addPredictedValue(self.archive[0].fitness_functions.values[m])
                    individual_evaluated.append(sh.copy())
                    self.archive.pop(0)
                    del(sh)
                else: #Generate a random individual. se puede mejorar usando la generación de un corte no existente en el train
                    #print("B")
                    individual = sch.Scheme(self.instances.ds, cuts={}, options=self.options)
                    surr_individual = individual.copy()
                    no_evaluations += individual.evaluate()
                    surr_individual.evaluate(self)
                    individual_evaluated.append(individual.copy())
                    for m in range(0, len(self.models)):
                        measures.addObservedValue(individual.fitness_functions.values[m])
                        measures.addPredictedValue(surr_individual.fitness_functions.values[m])
            #self.archive.clear()
            self.prediction_error = self.getPredictionError(measure=measures)
                   
        return individual_evaluated, no_evaluations
    
    def prune_train(self):
        if self.instances.population.size > self.train_size:
            randomidx = random.sample(range(0, self.instances.population.size), self.train_size)
            #pop_aux = pop.Population(self.instances.population.ds,pop_size=0,options=self.options)
            individuals = np.array(self.instances.population.individuals)
            #pop_aux.addIndividuals(individuals=individuals)
            self.instances.population.setIndividuals(individuals=individuals[randomidx])
    
    def UpdateAndTrain(self, individuals_evaluated=None):
        if individuals_evaluated:
            for f in individuals_evaluated:
                self.instances.population.add_individual(f)
                
            if self.options.ue == 1: #Recorto el conjunto de entrenamiento para mantener su tamaño
                self.instances.FastNonDominatedSort()
                self.instances.get_new_population()
                    
        
        # measures = RegressionMeasures()
            
        # if self.update_strategy == 1: #actualización usando el frente de pareto
        #     '''Así se hizo en el primer año'''
        #     if front:
        #         for f in front:
        #             self.instances.population.add_individual(f)
        #         self.instances.FastNonDominatedSort()
        #         self.instances.get_new_population()
                
                
        # if self.update_strategy == 2: #Número de individuos aleatorios
        #     measures = RegressionMeasures()
        #     for i in range(0, self.ind_upd):
        #         individual = sch.Scheme(self.instances.ds, cuts={}, idfunctionsconf=self.instances.population.individuals[0].idfunctionsconf)
        #         surr_individual = individual.copy()
        #         no_evaluations += individual.evaluate()
        #         surr_individual.evaluate(self)
        #         self.instances.population.add_individual(individual.copy())
        #         for m in range(0, len(self.models)):
        #             measures.addObservedValue(individual.fitness_functions.values[m])
        #             measures.addPredictedValue(surr_individual.fitness_functions.values[m])
        #         #del(individual)
        #     self.prediction_error = self.getPredictionError(measure=measures, type=self.evaluation_type)
                    
        # if self.update_strategy == 3:
        #     ''' Cuantos esquemas se ingresaran en el train? '''
        #     measures = RegressionMeasures()
        #     for i in range(0, self.ind_upd):
        #         #print("Archive", len(self.archive))
        #         if self.archive:
        #             #print("Copy:", i)
        #             #ind = randint(0, len(self.archive))
        #             sh = self.archive[0].copy()
        #             no_evaluations += sh.evaluate()
        #             for m in range(0, len(self.models)):
        #                 measures.addObservedValue(sh.fitness_functions.values[m])
        #                 measures.addPredictedValue(self.archive[0].fitness_functions.values[m])
        #             self.instances.population.add_individual(sh.copy())
        #             self.archive.pop(0)
        #             del(sh)
        #         else: #Generate a random individual. se puede mejorar usando la generación de un corte no existente en el train
        #             #print("Random:", i)
        #             individual = sch.Scheme(self.instances.ds, cuts={}, idfunctionsconf=self.instances.population.individuals[0].idfunctionsconf)
        #             surr_individual = individual.copy()
        #             no_evaluations += individual.evaluate()
        #             surr_individual.evaluate(self)
        #             self.instances.population.add_individual(individual.copy())
        #             for m in range(0, len(self.models)):
        #                 measures.addObservedValue(individual.fitness_functions.values[m])
        #                 measures.addPredictedValue(surr_individual.fitness_functions.values[m])
        #             #del(individual)
        #     self.prediction_error = self.getPredictionError(measure=measures, type=self.evaluation_type)
            
        #print("Cohesion: ",self.instances.population.cohesion_by_words())
            
        self.train()
        #return no_evaluations
        
    def train(self):
        self.models = []
        #deleted = self.delete_duplicated(self.instances.population.individuals)
        #self.instances.population.setIndividuals(deleted)
        #print(len(deleted), self.instances.population.size)
        self.prune_train()
        self.train_set, self.classes = self.instances.population.to_train_set()
        #print(self.instances.population.size)
        self.training_number += 1
        
        if self.options.model == 0: #KNN DTW, antes era 2
            self.needsFilled = False
            #print("Model:", self.model_type, ", k:", self.k, ", Metric:", self.metric, ", Optimizer:", self.optimizer, ", Window:", self.window, ", Update_strategy: ", self.update_strategy)
            #self.update_technique = 1 
            for i in range(0,len(self.classes[0])):
                knn = KNN(options=self.options)
                knn.fit(x=self.train_set, y=self.classes[:,i])
                self.models.append(knn.copy())
        
        if self.options.model == 1: #RBF
            #print("Model:", self.model_type, ", k:", self.k, ", Update_strategy", self.update_strategy)
            #self.needsFilled = True
            #self.update_technique = 2
            rbf_aux = RBF(options=self.options)
            #print(self.train_set)
            r = rbf_aux._compute_r(self.train_set)
            for i in range(0,len(self.classes[0])):
                rbf = RBF(options=self.options)
                rbf.fit(x=self.train_set, y=self.classes[:,i], r=r)
                self.models.append(rbf.copy())
                
         #if self.model_type == 6: #SVR GAK
        if self.options.model == 2: #SVR GAK
            #self.needsFilled = False      
            #print("Model:", self.model_type, ", Kernel:", self.k, ", Update_strategy", self.update_strategy)
            #self.update_technique = 2      
            for i in range(0,len(self.classes[0])):
                svr = SVR(options=self.options)
                svr.fit(self.train_set, np.array(self.classes[:,i]))
                self.models.append(svr.copy())
            #print(len(self.models))
            
                
        """ if self.model_type == 2: #KNN DTW, este fue el primero que uso en el primer año, con self.update_techinque = 1
            self.needsFilled = False
            print("Model:", self.model_type)
            self.update_techinque = 1 
            for i in range(0,len(self.classes[0])):
                best_knnlbdtw = KNN(n_neighbors=self.k, is_regression=True,  metric="dtw_fast", window=self.window)
                best_knnlbdtw.fit(self.train_set, self.classes[:,i])
                self.models.append(best_knnlbdtw.copy())
                
        if self.model_type == 3: #KNN LB DTW
            print("Model:", self.model_type)
            self.needsFilled = False
            self.update_techinque = 2
            for i in range(0,len(self.classes[0])):
                best_knnlbdtw = KNN(n_neighbors=self.k, is_regression=True,  metric="dtw_fast", optimizer="lb_kim", window=self.window)
                best_knnlbdtw.fit(self.train_set, self.classes[:,i])
                self.models.append(best_knnlbdtw.copy())
        
        if self.model_type == 4: #KNN soft_dtw
            self.needsFilled = False
            self.update_techinque = 2
            for i in range(0,len(self.classes[0])):
                best_knnlbdtw = KNN(n_neighbors=self.k, is_regression=True, metric="soft_dtw", window=self.window)
                best_knnlbdtw.fit(self.train_set, self.classes[:,i])
                self.models.append(best_knnlbdtw.copy())
        
        if self.model_type == 5: #KNN By Words
            self.needsFilled = False
            print("Model:", self.model_type)
            self.update_techinque = 3
            for i in range(0,len(self.classes[0])):
                best_knnlbdtw = KNN(n_neighbors=self.k, is_regression=True,  metric="dtw_fast", optimizer="by_words", window=self.window)
                best_knnlbdtw.fit(self.train_set, self.classes[:,i])
                self.models.append(best_knnlbdtw.copy()) """
            
        """ #if self.model_type == 6: #SVR GAK
        if self.options.model == 3: #SVR GAK
            #self.needsFilled = False      
            #self.update_strategy = 2      
            for i in range(0,len(self.classes[0])):
                svr = SVRModel(kernel="gak")
                svr.fit(self.train_set, self.classes[:,i])
                self.models.append(svr.copy())
            print(len(self.models)) """
            
        
            
