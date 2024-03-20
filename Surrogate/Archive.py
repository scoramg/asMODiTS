import eMODiTS.Scheme as sch
import random
from Utils.utils import delete_duplicated
import numpy as np

class Archive:
    def __init__(self, options=None):
        self.options = options
        self.data = []
    
    @property
    def size(self):
        return len(self.data)
    
    @property
    def is_empty(self):
        if len(self.data) > 0:
            return False
        else:
            return True
    
    @property
    def max_size(self):
        return int(round(self.options.ps/4,0))
    
    def add(self, data):
        self.data.append(data.copy())
        archive_aux = np.array(delete_duplicated(data=self.data))
        if len(archive_aux) > self.max_size:
            randomidx = random.sample(range(0, len(archive_aux)), self.max_size)
            self.set_data(archive_aux[randomidx])
            
    def set_data(self, data):
        self.data = np.array(data).tolist().copy()
        
    def create_checkpoint(self):
        individuals_cuts = []
        for i in range(0,self.size):
            #print("Archive.create_checkpoint.data[i][0]:", self.data[i].cuts)
            individuals_cuts.append(self.data[i].cuts.copy())
        return individuals_cuts
    
    def export_matlab(self):
        data = {}
        fitness = []
        surrogates = []
        isEvaluatedOriginal = []
        isEvaluatedSurrogate = []
        for i in range(0,self.size):
            #print("Archive.export_matlab.data[i]:",self.data[i])
            cuts, fits, surrs, isOriginal, isSurrogate = self.data[i].matlab_format()
            data["FrontIndividual"+str(i)] = cuts
            surrogates.append(surrs)
            fitness.append(fits)
            isEvaluatedOriginal.append(isOriginal)
            isEvaluatedSurrogate.append(isSurrogate)
        data["FrontFitness"] = fitness
        data["SurrogateFrontFitness"] = surrogates
        data["IsEvaluatedOriginalAccumulated"] = isEvaluatedOriginal
        data["IsEvaluatedSurrAccumulated"] = isEvaluatedSurrogate
        return data
    
    def restore(self, ds, checkpoint):
        for i in range(0, len(checkpoint)):
            self.add(data=sch.Scheme(ds=ds, cuts=checkpoint[i], options=self.options))
