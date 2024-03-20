import abc, math
from abc import ABC
import numpy as np
from DistanceMeasures.dtw import DTW
from DistanceMeasures.gak import GAK
from DistanceMeasures.tga import TGA
from Surrogate.Archive import Archive
from random import sample
from eMODiTS.Population import Population


class ModelBase(ABC):
    
    def __init__(self, id_model, options=None):
        self.training_set = None
        self.id_model = id_model
        self.options = options
        self.training_number = 0
        self.is_fitted = False
        self.is_trained = False
        self.gen_upd = math.floor(self.options.g/self.options.model[self.id_model].gu)
        self.factor_act = math.floor(self.options.g/self.options.model[self.id_model].gu)
        self.archive = Archive(options=options)
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        
    @property
    def class_name(self):
        return self.__class__.__name__
        
    @property
    def dist_metric(self):
        return self.options.model[self.id_model].dist_metric
        
    @property
    def distance_measure(self):
        if self.options.model[self.id_model].dist_metric == 'dtw':
            return DTW(options=self.options.model[self.id_model])
        if self.options.model[self.id_model].dist_metric == 'gak':
            return GAK(n_jobs=-1)
        if self.options.model[self.id_model].dist_metric == 'tga':
            return TGA(options=self.options.model[self.id_model])
    
    @property
    def dist_threshold(self):
        return np.mean(self.matrix_distances)
    
    @property
    def higher_distances_indexes(self):
        d_minimal = np.mean(self.matrix_distances, axis = 1)
        d_minimal_threshold = np.where(d_minimal > self.dist_threshold, d_minimal, np.nan)
        ordered = (-d_minimal_threshold).argsort().ravel()
        return ordered[np.argwhere(~np.isnan(d_minimal_threshold[ordered]))].flatten()
        
    @property    
    def matrix_distances(self):
        return self._matrix_distances
    
    @matrix_distances.setter
    def matrix_distances(self, matrix_distances):
        self._matrix_distances = matrix_distances
        
    @property    
    def normalized_matrix_distance(self):
        return self._normalized_matrix_distance
    
    @normalized_matrix_distance.setter
    def normalized_matrix_distance(self, normalized_matrix_distance):
        self._normalized_matrix_distance = normalized_matrix_distance
        
    @abc.abstractmethod
    def get_name(self):
        raise NotImplementedError("Subclase debe implementar el método abstracto")
    
    @abc.abstractmethod
    def train(self):
        raise NotImplementedError("Subclase debe implementar el método abstracto")
    
    @abc.abstractmethod
    def predict(self, x):
        raise NotImplementedError("Subclase debe implementar el método abstracto")
    
    @abc.abstractmethod
    def export_matlab(self):
        raise NotImplementedError("Subclase debe implementar el método abstracto")
    
    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError("Subclase debe implementar el método abstracto")
            
    def _update_archive(self, *args):
        updated = 0
        ratio_updated = 0
        evaluations = 0
        if self.archive.is_empty:
            ratio_updated, no_eval = self._update_random(args[0])
            evaluations += no_eval
        else:
            data = []
            if self.archive.size < self.options.batch_update:
                data = self.archive.data.copy()
                for j in range(0, self.options.batch_update-self.archive.size):
                    data.append(args[0][j])
            else:
                indexes = sample(range(0,self.archive.size), self.options.batch_update)
                for j in indexes:
                    data.append(self.archive.data[j])
                self.archive.set_data(np.delete(self.archive.data, indexes)) 
                
            for i in range(0,len(data)):
                ind = data[i]
                evaluations += ind.evaluate()
                error = ind.prediction_measures(id_model=self.id_model)
                print("ModelUpdate.Error:", error, "Error threshold:", self.options.error_t)
                if error > self.options.error_t:
                    self.training_set.add_individual(ind)
                    updated += 1
            ratio_updated = float(updated/self.options.batch_update)
        return ratio_updated, evaluations
            
    def _update_random(self, *args):
        updated = 0
        evaluations = 0
        data = args[0]
        for i in range(0,self.options.batch_update):
            ind = data[i]
            evaluations += ind.evaluate()
            error = ind.prediction_measures(id_model=self.id_model)
            if error > self.options.error_t:
                self.training_set.add_individual(ind)
                updated += 1
        return float(updated/self.options.batch_update), evaluations
    
    def fit(self, training_set):
        self.training_set = training_set.copy()
        self.is_fitted = True
    
    def update(self, params=None): 
        updated , no_eval= self._update_archive(params["archive"])
        if updated > 0:
            self.train()
        return updated, no_eval

    def insert_archive(self, **kwargs):
        idxs = self.higher_distances_indexes
        individuals = np.array(kwargs['individuals'])
        for i in range(len(idxs)):
            self.archive.add(individuals[idxs[i].astype(int)])
    
    def restore(self, ds, checkpoint):
        training_set = Population(_ds=ds, options=self.options)
        training_set.restore(checkpoint["training_set"])
        self.fit(training_set)
        self.train()
        self.archive.restore(ds=ds,checkpoint=checkpoint["archive"])
        self.gen_upd = checkpoint["gen_upd"]
        self.factor_act = checkpoint["factor_act"]
        self.is_trained = checkpoint["is_trained"] 
        self.is_fitted = checkpoint["is_fitted"]
        self.training_number = checkpoint["training_number"]