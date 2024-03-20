import itertools, pickle, math, functools#, lzma, bz2, gzip
import eMODiTS.Population as pop
from Utils.utils import compare_by_rank_crowdistance, delete_files_pattern, find_file, find_last_file
import numpy as np
import random, time
from RegressionMeasures.regression_measures import RegressionMeasures
import cProfile, pstats
from pstats import SortKey 
from EvolutionaryMethods.pareto_front import ParetoFront

class NSGA2:
    def __init__(self,**kwargs):
        
        self.__dict__.update(kwargs)
        self.fronts = {}
        self.fronts[0] = ParetoFront(ds=self.ds, options=self.options)
        self.population = pop.Population(_ds=self.ds, options=self.options)
        self.no_evaluations = 0
        self.surrogate_errors = {}
        self._gen_upd = 0
        self._iu = 0
        self._factor_act = 0
        
    @property
    def size(self):
        if not hasattr(self,'max_size'):
            return self.options.ps
        else:
            return self.max_size
    
    @property
    def ds(self):
        if not hasattr(self,'_ds'):
            return None
        else:
            return self._ds
    
    @ds.setter 
    def ds(self, ds):    
        self._ds = ds.copy()
        self.population.ds = self.ds.copy()
    
    @property
    def surrogate_models(self):
        if not hasattr(self,'_surrogate_models'):
            return None
        else:
            return self._surrogate_models
        
    @property
    def options(self):
        if not hasattr(self,'_options'):
            return None
        else:    
            return self._options
        
    def __del__(self):
        del(self.fronts)
        del(self._ds)
        del(self._surrogate_models)
        del(self.population)
        
    def set_population(self, population):
        self.population = population.copy()
        
    def non_ranked(self):
        ranked = True
        for ind in self.population.individuals:
            if ind.rank == -1:
                ranked = False
                break
        return ranked
    
    def get_minimum_domination_count(self):
        count_min = 10000000
        for ind in self.population.individuals:
            if ind.domination_count < count_min and ind.domination_count > 0:
                count_min = ind.domination_count
        return count_min
    
    def FastNonDominatedSort(self):  
        self.fronts = {} 
        self.fronts[0] = ParetoFront(ds=self.ds, options=self.options) 
        for p in range(0,self.population.size):
            self.population.individuals[p].dominated_set = []
            self.population.individuals[p].domination_count = 0
            self.population.individuals[p].rank = -1
            for q in range(0,self.population.size):
                if self.population.individuals[p].dominates(self.population.individuals[q]):
                    self.population.individuals[p].dominated_set.append(self.population.individuals[q]) #S
                elif self.population.individuals[q].dominates(self.population.individuals[p]):
                    self.population.individuals[p].domination_count += 1
            if self.population.individuals[p].domination_count == 0:
                self.population.individuals[p].rank = 0
                self.fronts[0].add(self.population.individuals[p])
            
        i = 0
        while self.fronts[i].size > 0:
            Q = ParetoFront(ds=self.ds, options=self.options)
            for j in range(0,self.fronts[i].size):
                for dom in self.fronts[i].get(j).dominated_set:
                    dom.domination_count -= 1
                    if dom.domination_count == 0:
                        dom.rank = i + 1
                        if dom not in Q.points.individuals:
                            Q.add(dom)
            i += 1
            self.fronts[i] = Q.copy()
            del(Q)
        
        if not self.non_ranked():
            domination_count = self.get_minimum_domination_count()
            print(domination_count)
       
    def get_crowding_distance(self):
        for f in range(0, len(self.fronts)):
            if self.fronts[f].size > 0:
                if self.fronts[f].size in [1,2]:
                    for i in range(0,self.fronts[f].size):
                        self.fronts[f].get(i).crowding_distance = math.inf
                else:
                    for m in range(0,self.fronts[f].get(0).fitness_functions.size):
                        self.fronts[f].individuals.sort(key=lambda x: x.fitness_functions.values[m], reverse=False)     
                        self.fronts[f].get(0).crowding_distance = math.inf
                        self.fronts[f].get(self.fronts[f].size-1).crowding_distance = math.inf
                        maximum = max(front.fitness_functions.values[m] for front in self.fronts[f].individuals)
                        minimum = min(front.fitness_functions.values[m] for front in self.fronts[f].individuals)
                        FmaxFminDiff = maximum-minimum
                        if FmaxFminDiff == 0:
                            FmaxFminDiff = 1
                        for k in range(1,self.fronts[f].size-1):
                            self.fronts[f].get(k).crowding_distance = self.fronts[f].get(k).crowding_distance + ((self.fronts[f].get(k+1).fitness_functions.values[m] - self.fronts[f].get(k-1).fitness_functions.values[m])/FmaxFminDiff)
     
    def get_new_population(self):
        new_population = pop.Population(_ds=self.population.ds, options=self.options)
        t=0
        for f in range(0, len(self.fronts)):
            if (t+self.fronts[f].size) < self.size:
                for i in range(0,self.fronts[f].size):
                    new_population.add_individual(self.fronts[f].get(i))
                    t+=1
            else:
                j=0
                last = self.fronts[f].individuals
                last.sort(key=functools.cmp_to_key(compare_by_rank_crowdistance))
                while t < self.size:
                    new_population.add_individual(last[j])
                    t+=1
                    j+=1
                break
        self.population = new_population.copy()
        del(new_population)
        
    def get_first_front_as_population(self):
        population = pop.Population(_ds=self.ds, options=self.options)
        for sc in self.fronts[0].individuals:
            population.add_individual(sc)
        return population
    
    def get_first_front(self, population=None, is_already_sorted=False):
        if population:
            self.set_population(population=population)
        if not is_already_sorted:
            self.FastNonDominatedSort()
        return self.fronts[0]    
    
    def print_first_front(self):
        for i in range(0, self.fronts.size):
            for j in range(0, self.fronts[i].size):
                print(i,self.fronts[i].get(j).fitness_functions.values)
    
    def run_generation(self, g, e):
        print("Model:", self.surrogate_models.name, " - Dataset:", self.ds.name," - Ejecución:", e," - Generación NSGA2:", g+1)
        self.FastNonDominatedSort()
        self.get_crowding_distance()
        parents = self.population.tournament_selection()
        offsprings = parents.crossover()
        offsprings.mutate()
        
        self.no_evaluations += offsprings.evaluate(self.surrogate_models)
        
        inds = random.sample(range(0,offsprings.size),self._iu)
        for ind in inds:
            self.no_evaluations += offsprings.individuals[ind].evaluate()
        
        self.population.join(offsprings)
        self.FastNonDominatedSort()
        self.get_crowding_distance()
        self.get_new_population()
        
        del(offsprings)
        del(parents)
    
    def restore(self, e, dir):
        generation, file = find_file(dir, include="checkpoint_e{ex}_g".format(ex=e))
        if generation > 0:
            try:
                checkpoint = pickle.load(open(dir+"/"+file, "rb" ))
                self.surrogate_models.restore(self.ds, checkpoint["surrogate_models"])
                self.population.restore(checkpoint["population"])
                self.no_evaluations = checkpoint["no_evaluations"]
                self.surrogate_errors = checkpoint["surrogate_errors"]
                self.fronts[0].restore(checkpoint["front"])
            except pickle.UnpicklingError as e:
                print("nsga2.Error: Corrupt checkpoint.")
                generation = -1
                file = None
        return generation
    
    def create_checkpoint(self, e, g, dirs):
        if self.options.cache:
            self.options.cache_data.save()
        delete_files_pattern(dirs["checkpoints"],"checkpoint_e*_g*")
        cp = dict(surrogate_models=self.surrogate_models.create_checkpoint(), 
                  population=self.population.get_individuals_cuts(), 
                  no_evaluations=self.no_evaluations, 
                  surrogate_errors=self.surrogate_errors,
                  front=self.fronts[0].points.get_individuals_cuts())
        no_digits = len(str(self.options.g))
        pickle.dump(cp, open(dirs["checkpoints"]+"checkpoint_e"+str(e)+"_g"+str(g).zfill(no_digits)+".pkl", "wb" ), protocol=pickle.HIGHEST_PROTOCOL)
            
    def _generate_random_individuals(self):
        aux = pop.Population(_ds=self.ds,pop_size=self.options.batch_update, options=self.options)
        self.no_evaluations += aux.evaluate(surrogate_models=self.surrogate_models)
        return aux.individuals.copy()
    
    def execute(self, e, dirs, population = None):
        if population:
            self.set_population(population=population)
        else:
            self.population.create(self.options.ps)
            self.no_evaluations += self.population.evaluate()
        
        self._iu = math.floor(self.options.iu * self.options.ps)
        
        self.surrogate_errors['Execution'] = {}
        for i in range(0,self.surrogate_models.no_models):
            self.surrogate_errors['Execution']['Model'+str(i)] = RegressionMeasures.init_measures_values()
        
        if self.options.checkpoints:
            g_ini = self.restore(e=e, dir=dirs["checkpoints"]) + 1
        else:
            g_ini = 0
            
        models = list(self.surrogate_errors['Execution'].keys())
        measures = list(list(self.surrogate_errors['Execution'].values())[0].keys())
        ids = list(itertools.product(np.arange(0,len(models)),np.arange(0,len(measures))))
        for g in range(g_ini,self.options.g):
            if self.options.profilers:
                pr = cProfile.Profile()
                pr.enable() 
            self.run_generation(g, e)
            predictions = self.population.prediction_power(surrogate_models=self.surrogate_models)
            inds = self._generate_random_individuals()
            included, no_eval = self.surrogate_models.update(g=g,front=self.fronts[0].individuals.copy(),random=inds.copy(),archive=inds.copy(),individuals=self.population.individuals.copy())
            self.no_evaluations += no_eval
            if included > 0:
                print("Model Updated")
                self.no_evaluations += self.population.evaluate(self.surrogate_models)
                self.no_evaluations += self.population.evaluate()
                    
            self.surrogate_errors['Generation'+str(g)] = predictions.copy()
            
            for i,j in ids:
                self.surrogate_errors['Execution'][models[i]][measures[j]] += predictions[models[i]][measures[j]]
                
            if self.options.checkpoints:
                self.create_checkpoint(e=e, g=g, dirs=dirs)
                
            if self.options.profilers:
                pr.disable()
                sortby = SortKey.CUMULATIVE
                ExportProfileToFile = dirs["profiler_e"+str(e)] + "ProfilerResults_e"+str(e)+"_g"+str(g)+"_"+self.surrogate_models.name+"_"+time.strftime("%Y-%m-%d-%H-%M-%S")+".txt"
                with open(ExportProfileToFile, 'w') as stream:
                    stats = pstats.Stats(pr, stream=stream).sort_stats(sortby)
                    stats.print_stats()
        
        for i,j in ids:
            self.surrogate_errors['Execution'][models[i]][measures[j]] = self.surrogate_errors['Execution'][models[i]][measures[j]] / self.options.g
