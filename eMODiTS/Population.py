import eMODiTS.Scheme as sch
#import Surrogate.Model as model
import math, random #, copy
from Utils.utils import compare_by_rank_crowdistance
import numpy
#import gc
from dtaidistance.dtw import distance_fast
from DistanceMeasures.dtw import DTW
from DistanceMeasures.tga import TGA
from DistanceMeasures.gak import GAK
from RegressionMeasures.regression_measures import RegressionMeasures

class Population:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
        self.individuals = []
        if hasattr(self,"pop_size"):
            self.create(pop_size=self.pop_size)
            
    @property
    def size(self):
        return len(self.individuals)
    
    @property
    def ds(self):
        return self._ds
    
    @ds.setter
    def ds(self, ds):
        self._ds = ds
    
    def __del__(self):
        del(self.individuals)
        if hasattr(self,"_ds"):
            del(self._ds)
        
    def __str__(self):
        s = ""
        for i in range(0,len(self.individuals)):
            s = s + str(self.individuals[i].cuts)+" \n"+str(self.individuals[i].fitness_functions.values)+"\n";
        return s
    
    def setIndividuals(self, individuals):
        self.individuals = []
        for ind in individuals:
            self.add_individual(ind)
    
    def addIndividuals(self, individuals):
        for ind in individuals:
            self.add_individual(ind)
                    
    def add_individual(self, scheme):
        self.individuals.append(scheme.copy())
        
    def create(self, pop_size):
        for i in range(0,pop_size):
            self.add_individual(sch.Scheme(ds=self.ds, options=self.options))
            
    def evaluate(self, surrogate_models=None): 
        no_eval = 0
        if not surrogate_models:
            for i in range(0,self.size):
                no_eval += self.individuals[i].evaluate()
                self.individuals[i].isEvaluatedOriginal = True
        else:   
            surrs = numpy.empty([self.size, surrogate_models.no_models], dtype=float)
            surrs[:] = numpy.nan
            for i in range(0, surrogate_models.no_models):
                train, _ = self.to_train_set(i)
                y_pred = surrogate_models.models[i].predict(train)
                surrs[:,i] = y_pred
            for j in range(0, self.size):
                if numpy.all(numpy.isnan(surrs[j,:])):
                    print("Population.evaluation: Entro")
                self.individuals[j].surrogate_values = surrs[j,:].tolist()
                self.individuals[j].fitness_functions.values = surrs[j,:].tolist()
                self.individuals[j].isEvaluatedSurrogate = True
            for i in range(0,surrogate_models.no_models):
                eval("surrogate_models.models[i].insert_"+self.options.model[i].ue)(individuals=self.individuals)
            
        return no_eval
    
    def crossover(self):
        offsprings = Population(_ds=self.ds, options=self.options)
        k = self.size-1
        for i in range(0,self.size):
            if i < k:
                if random.random() <= self.options.pc:
                    off1, off2 = self.individuals[i].crossover(self.individuals[k])
                    offsprings.add_individual(off1)
                    offsprings.add_individual(off2)
                else:
                    off1 = self.individuals[i].copy()
                    off1.clear_ff()
                    off2 = self.individuals[k].copy()
                    off2.clear_ff()
                    offsprings.add_individual(off1)
                    offsprings.add_individual(off2)
            k-=1
        return offsprings
    
    def mutate(self):
        for i in range(0,self.size):
            self.individuals[i].mutate()
    
    def copy(self):
        if hasattr(self,"_ds"):
            mycopy = Population(_ds=self.ds, options=self.options)
        else:
            mycopy = Population(options=self.options)
        for i in range(0,len(self.individuals)):
            mycopy.add_individual(self.individuals[i].copy())
        return mycopy
    
    def join(self, other):
        for i in range(0, other.size):
            self.add_individual(other.individuals[i])
    
    def tournament_selection(self):
        parents = Population(_ds=self.ds, options=self.options)
        no_opponents = math.floor(self.size * 0.1)
        victories = {}
        for i in range(0,self.size):
            victories[i] = 0
            opponents = []
            opponents.append(i)
            for op in range(0,no_opponents):
                j = random.choice([r for r in range(0,self.size) if r not in opponents])
                if compare_by_rank_crowdistance(self.individuals[i], self.individuals[j]) < 0:
                    victories[i] += 1
                opponents.append(j)
        victories = {k: v for k, v in sorted(victories.items(), key=lambda item: item[1], reverse=True)}
        for i in list(victories.keys())[:self.size]:
            parents.add_individual(self.individuals[i])
        del(victories)
        return parents
    
    def to_train_set(self, id_model):
        train = []
        ff = []
        for i in range(0,self.size):
            data, ff_values = self.individuals[i].to_vector(id_model)
            train.append(data)
            ff.append(ff_values)
        
        return numpy.array(train, dtype=object), numpy.array(ff)
        
    def export_matlab(self, isAccumulated = True):
        data = {}
        fitness = []
        surrogates = []
        isOriginal = []
        isSurrogate = []
        for i in range(0,len(self.individuals)):
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
    
    def cuts_probabilities(self):
        prob_cuts = {}
        for i in range(0, len(self.individuals)):
            if prob_cuts.get(len(self.individuals[i].cuts)) is not None:
                prob_cuts[len(self.individuals[i].cuts)] += 1
            else:
                prob_cuts[len(self.individuals[i].cuts)] = 1
        for key in prob_cuts.keys():
            prob_cuts[key] /= len(prob_cuts.keys())
        return prob_cuts
    
    def cohesion(self):
        suma = 0
        for i in range(0,len(self.individuals)):
            for j in range(0,len(self.individuals)):
                data1, ff_values1 = self.individuals[i].to_vector()
                data2, ff_values2 = self.individuals[j].to_vector()
                suma += distance_fast(numpy.array(data1),  numpy.array(data2), use_pruning=True)
        return suma/len(self.individuals)
    
    def cohesion_by_words(self):
        dic = self.cuts_probabilities()
        mvalue = len(dic.keys())
        return mvalue
        
        
    def fitness_values(self): # 0: eMODiTS, 1: eMODiTS Surrogated
        data = {}
        data['original'] = []
        data['surrogate'] = []
        data['inserted'] = 0
        for ind in self.individuals:
            if ind.is_prediction_measurable:
                data['original'].append(ind.fitness_functions.values.copy())
                data['surrogate'].append(ind.surrogate_values.copy())
                data['inserted'] += 1
        data['original'] = numpy.array(data['original'])
        data['surrogate'] = numpy.array(data['surrogate'])
        return data
            
    def prediction_power(self, surrogate_models):
        fitness = self.fitness_values()
        obs = fitness['original']
        surr = fitness['surrogate']
        data = {}
        
        for i in range(0,surrogate_models.no_models):
            if fitness['inserted'] > 0:
                try:
                    measures = RegressionMeasures(observed=obs[:,i], predicted=surr[:,i])
                except IndexError:
                    exit(1)
            else:
                measures = RegressionMeasures()
            measures.compute()
            data['Model'+str(i)] = measures.export_matlab()
        return data
            
    def load_matlab(self, data):
        fitness = data['FrontFitness']
        surrogates = data['SurrogateFrontFitness']
        for i in range(0,len(fitness)):
            name = "FrontIndividual"+str(i)
            individual = sch.Scheme(ds=self.ds, options=self.options)
            individual.reset()
            individual.load_from_lists(cuts=data[name], ff=fitness[i], surr=surrogates[i])
            self.add_individual(individual)
            
    def get_individuals_cuts(self):
        individuals_cuts = []
        for i in range(0,self.size):
            ind = {}
            ind["IndividualCuts"] = self.individuals[i].cuts.copy()
            ind["FitnessValues"] = self.individuals[i].fitness_functions.values.copy()
            ind["SurrogateValues"] = self.individuals[i].surrogate_values.copy()
            ind["isEvaluatedOriginal"] = self.individuals[i].isEvaluatedOriginal
            ind["isEvaluatedSurrogate"] = self.individuals[i].isEvaluatedSurrogate
            individuals_cuts.append(ind)
        return individuals_cuts
    
    def restore(self, individuals_cuts):
        for i in range(0,len(individuals_cuts)):
            ind = sch.Scheme(ds=self.ds, cuts=individuals_cuts[i]["IndividualCuts"], options=self.options)
            ind.fitness_functions.values = individuals_cuts[i]["FitnessValues"]
            ind.surrogate_values = individuals_cuts[i]["SurrogateValues"]
            ind.isEvaluatedOriginal = individuals_cuts[i]["isEvaluatedOriginal"]
            ind.isEvaluatedSurrogate = individuals_cuts[i]["isEvaluatedSurrogate"]
            self.add_individual(ind)
            
        