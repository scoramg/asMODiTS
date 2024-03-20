from eMODiTS.Population import Population

class ParetoFront:
    def __init__(self, ds=None, options = None):
        self._ds = ds
        self.options = options
        self.points = Population(_ds=self.ds, options=options)
        
    @property
    def size(self):
        return self.points.size
    
    @property
    def individuals(self):
        return self.points.individuals
    
    @property
    def ds(self):
        return self._ds
    
    @ds.setter
    def ds(self, ds):
        self._ds = ds
        self.points.ds = ds
        
    def add(self, ind):
        self.points.add_individual(ind)
    
    def get(self, i):
        return self.points.individuals[i]
        
    def addIndividualsForFront(self, front):
        for i in range(0, front.size):
            self.add(front.individuals[i])
        
    def export_matlab(self, isAccumulated = True):
        data = {}
        fitness = []
        surrogates = []
        isOriginal = []
        isSurrogate = []
        for i in range(0,self.size):
            cuts, fits, surrs, isorig, issurr = self.individuals[i].matlab_format()
            data["FrontIndividual"+str(i)] = cuts
            surrogates.append(surrs)
            fitness.append(fits)
            isOriginal.append(isorig)
            isSurrogate.append(issurr)
        if isAccumulated:
            data["AccumulatedFrontFitness"] = fitness
            data["SurrugateAccumulatedFront"] = surrogates
            data["IsEvaluatedOriginalAccumulated"] = isOriginal
            data["IsEvaluatedSurrAccumulated"] = isSurrogate
        else:
            data["FrontFitness"] = fitness
            data["SurrogateFrontFitness"] = surrogates
            data["IsEvaluatedOriginal"] = isOriginal
            data["IsEvaluatedSurr"] = isSurrogate
        return data
        
    def load(self, data):
        self.points = Population(_ds=self.ds, options=self.options)
        self.points.load_matlab(data=data)
    
    def get_fronts_checkpoint(self):
        return self.points.get_individuals_cuts()
    
    def restore(self,individuals_cuts):
        self.points.restore(individuals_cuts)
        
    def copy(self):
        pf = ParetoFront(ds=self.ds, options=self.options)
        pf.points = self.points.copy()
        return pf
                
                
                
        