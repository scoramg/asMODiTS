import itertools
import os, random, sys#, math
import collections#, gc
import numpy as np
import pandas as pd
import scipy.io
import pandas as pd
#from sklearn.model_selection import cross_val_score
from sklearn import tree
#from sklearn import metrics
import graphviz
import dot2tex
#from DistanceMeasures.dtw import DTW
#from DistanceMeasures.gak import GAK
from RegressionMeasures.regression_measures import RegressionMeasures

sys.path.append(os.path.abspath(os.path.join('.', 'src')))

import Settings as conf

sys.path.append(os.path.abspath(os.path.join('.', 'src/Functions')))
import Functions.confusion_matrix as cm
import Functions.fitness_functions as ff
from Utils.utils import normalize_matrix, to_paired_value, separate_data_target, minmax_normalize, minmax_normalize_number
from classifier import Classifier
from Datasets.dataset import Dataset
from statistics import mean, pstdev

#from memory_profiler import profile

def transformation(number, intervals, j, clase):
    if j>0:
        idx_list = [i+1 for i,interval in enumerate(intervals) if interval[0] <= number < interval[1]]
        idx = 1 if not idx_list and number < intervals[0][0] else len(intervals) if not idx_list and number > intervals[len(intervals)-1][0] else idx_list[0]
        return idx
    else:
        return clase

class Scheme:
        
    def __init__(self, **kwargs):
        #self.cuts = cuts.copy()
        self.__dict__.update(kwargs)
        self.error_rate = -1.0
        #self.options = options
        
        self.rank = -1
        self.crowding_distance = 0
        self.domination_count = 0
        self.dominated_set = set()
        
        #self.ds = ds
        self.ds_discrete_ints = []
        self.ds_discrete_strings = []
        self.ds_discrete_reconstructed = []
        self.fitness_functions = ff.FitnessFunction(self.options.ff)
        self.confusion_matrix = cm.ConfusionMatrix(str_discrete=[], bd=self.ds)
        
        self.classifier = Classifier(clases=self.ds.clases,clf_id=0)
        self.DecisionTreeGraph = None
        self.DecisionTreeGraphText = ""
        self.statistic_rates = None
        
        self.surrogate_values = [np.nan]*self.fitness_functions.no_functions
        self.isEvaluatedOriginal = False
        self.isEvaluatedSurrogate = False
        
        self._max_array_size = int(conf.MAX_NUMBER_OF_WORD_CUTS*self.ds.dimensions[1])*(conf.MAX_NUMBER_OF_ALPHABET_CUTS-1)
        if not hasattr(self,"cuts"):
            if not hasattr(self,"filename"):
                self.random_load()
            else:
                self.load_from_matlab(self.filename)
    
    @property
    def is_prediction_measurable(self):
        if self.isEvaluatedSurrogate and self.isEvaluatedOriginal:
            return True
        else:
            return False
    
    def reset(self):
        self.cuts = {}
        self.error_rate = -1.0
        self.rank = -1
        self.crowding_distance = 0
        #self.front = -1
        self.domination_count = 0
        self.dominated_set = set()
        self.ds_discrete_ints = []
        self.ds_discrete_strings = []
        self.ds_discrete_reconstructed = []
        self.fitness_functions = ff.FitnessFunction(self.options.ff)
        self.confusion_matrix = cm.ConfusionMatrix(str_discrete=[], bd=self.ds)
        self.surrogate_values = [np.nan] * self.fitness_functions.no_functions
        self.isEvaluatedOriginal = False
        self.isEvaluatedSurrogate = False
    
    def clear_ff(self):
        self.fitness_functions = ff.FitnessFunction(self.options.ff)
    
    def create_alphs_cuts(self):
        num_alph = random.randint(conf.MIN_NUMBER_OF_ALPHABET_CUTS, conf.MAX_NUMBER_OF_ALPHABET_CUTS-1)
        alphs = {random.uniform(self.ds.limites[0], self.ds.limites[1]) for i in range(num_alph)}
        alphs_cuts = to_paired_value(sorted(list(alphs.union({self.ds.limites[0], self.ds.limites[1]}))))
        return alphs_cuts
        
    def random_load(self):
        num_cuts = random.randint(conf.MIN_NUMBER_OF_WORD_CUTS, int(conf.MAX_NUMBER_OF_WORD_CUTS*self.ds.dimensions[1]))
        #word_cuts = set()
        wordcuts = {random.randint(1, self.ds.dimensions[1]-1) for i in range(num_cuts)}
        #for _ in range(0, num_cuts):
        #    word_cuts.add(random.randint(1, self.ds.dimensions[1]-1))
        word_cuts = to_paired_value(sorted(list(wordcuts.union({1,self.ds.dimensions[1]-1}))))
        
        self.cuts = {str(word_cuts[i]): self.create_alphs_cuts() for i in range(0,len(word_cuts))}
        
        """ for my_key in word_cuts:
            num_alph = random.randint(conf.MIN_NUMBER_OF_ALPHABET_CUTS, conf.MAX_NUMBER_OF_ALPHABET_CUTS-1)
            #alphs = set()
            #for _ in range(0, num_alph):
            #    alphs.add(random.uniform(self.ds.limites[0], self.ds.limites[1]))
            alphs = {random.uniform(self.ds.limites[0], self.ds.limites[1]) for i in range(num_alph)}
            alphs_cuts = to_paired_value(sorted(list(alphs.union({self.ds.limites[0], self.ds.limites[1]}))))
            self.cuts[str(my_key)] = alphs_cuts """
            
    def load_from_matlab(self, filename):
        sch = scipy.io.loadmat(filename)
        wordcuts = set(sch['cuts'][:,0])
        word_cuts = to_paired_value(sorted(list(wordcuts.union({1,self.ds.dimensions[1]-1}))))
        alphs = sch['cuts'][0,1:len(sch['cuts'])]
        for i in range(0,len(word_cuts)):
            alphs_cuts = sch['cuts'][i,1:len(sch['cuts'][i])]
            alphs = set(alphs_cuts[~np.isnan(alphs_cuts)])
            alphscuts = to_paired_value(sorted(list(alphs.union({self.ds.limites[0], self.ds.limites[1]+1}))))
            self.cuts[str(word_cuts[i])] = list(alphscuts)
            
    def load_from_lists(self, cuts, ff=None, surr=None):
        wordcuts = set(cuts[:,0])
        word_cuts = to_paired_value(sorted(list(wordcuts.union({1,self.ds.dimensions[1]-1}))))
        for i in range(0,len(word_cuts)):
            alphs_cuts = cuts[i,1:len(cuts[i])]
            alphs = set(alphs_cuts[~np.isnan(alphs_cuts)])
            alphscuts = to_paired_value(sorted(list(alphs.union({self.ds.limites[0], self.ds.limites[1]+1}))))
            self.cuts[str(word_cuts[i])] = list(alphscuts)
        if len(ff):
            self.fitness_functions.values = ff.copy()
        if len(surr):
            self.surrogate_values = surr.copy()
            
    def extract_data(self):
        inits = []
        ends = []
        alphs = []
        cuts = list(self.cuts.keys())
        for cut in cuts:
            alphs.append(self.cuts[str(cut)])
            interval = cut.replace("["," ").replace("]"," ")
            interval = interval.split(",")
            inits.append(int(float(interval[0])))
            ends.append(int(float(interval[1])))
        return inits, ends, alphs
    
    def cutdiffs(self):
        diffs = []
        for wordcuts, alph_cuts in self.cuts.items():
            interval = wordcuts.replace("["," ").replace("]"," ")
            interval = interval.split(",")
            u = int(float(interval[1]))
            l = int(float(interval[0]))
            if l == 1:
                l-=1
            diffs.append(u - l)
        return diffs
    
    """ def _discretize_int(self, data): #Borrar
        discrete = []
        inits, ends, alphs = self.extract_data()
        discrete = []
        for i in range(0,len(data)):
            row_discr = []
            row_discr.append(int(data[i,0]))
            for j in range(0,len(inits)):
                media = data[i,range(inits[j],ends[j])].mean()
                for k in range(0,len(alphs[j])):
                    if alphs[j][k][0] <= media < alphs[j][k][1]:
                        row_discr.append(k+1)
                if media >= alphs[j][len(alphs[j])-1][1]:
                    row_discr.append(len(alphs[j]))
                if media < alphs[j][0][0]:
                    row_discr.append(1)
            discrete.append(row_discr)
        return np.array(discrete) """

    def discretize_int(self, data):
        inits, ends, alphs = self.extract_data()
        self.ds_discrete_ints = np.array([[transformation(data[r,inits[j-1]:ends[j-1]].mean(), alphs[j-1], j, int(data[r,0])) for j in range(0,len(inits)+1)] for r in range(0,len(data))])
            
    def discretize(self):
        self.discretize_int(self.ds.data)
        self.reconstruct()
        
        #strings = []
        #for i in range(0,len(self.ds_discrete_ints)):
        #    string = ""
        #    for j in range(1,len(self.ds_discrete_ints[i,:])):
        #        string += conf.LETTERS[self.ds_discrete_ints[i,j]]
        #    strings.append([self.ds_discrete_ints[i,0],string])
        #self.ds_discrete_strings = np.array(strings)
        discrete2 = self.ds_discrete_ints.copy()
        discrete2[:,1:] += 64
        self.ds_discrete_strings = np.array([[discrete2[i,0], ''.join(map(chr,discrete2[i,1:].flatten()))] for i in range(len(discrete2))])
        self.confusion_matrix.create(self.ds_discrete_strings)
    
    """ def _reconstruct(self): #Borrar
        #self.ds_discrete_ints = np.array(self.ds_discrete_ints)
        diffs = self.cutdiffs()
        reconstructed = np.empty([len(self.ds.data), len(self.ds.data[0,:])])
        for i in range(0,len(self.ds_discrete_ints)):
            reconstructed_row = []
            try:
                reconstructed[i,0] = self.ds_discrete_ints[i,0]
            except IndexError:
                df = pd.DataFrame(self.ds_discrete_ints).T
                df.to_excel(excel_writer = "/Users/scoramg/Dropbox/Escolaridad/Postdoctorado/python/errors.xlsx")
                print(len(self.ds_discrete_ints),len(self.ds.ds_discrete_ints[0,:]))
            for j in range(1,len(self.ds_discrete_ints[i,:])):
                reconstructed_row = reconstructed_row + list(itertools.repeat(self.ds_discrete_ints[i,j], diffs[j-1]))
            if len(reconstructed[i,1:]) != len(np.array(reconstructed_row)):
                print("i",i)
                print("len(reconstructed[i,1:]):", len(reconstructed[i,1:]), "len(np.array(reconstructed_row)):", len(np.array(reconstructed_row)))
            np.copyto(reconstructed[i,1:],np.array(reconstructed_row))  
        
        self.ds_discrete_reconstructed = normalize_matrix(reconstructed)   """
    
    def reconstruct(self):
        #self.ds_discrete_ints = np.array(self.ds_discrete_ints)
        diffs = self.cutdiffs()
        #i=0
        #j=0
        #print(self.ds_discrete_ints, diffs[0])
        #print(list(itertools.chain.from_iterable(itertools.repeat(self.ds_discrete_ints[j,i], diffs[i]))))
        
        diffs.insert(0,1) # Inserta al inicio de las diferencias un 1, para que se repita 1 vez la clase
        stats = [[self.ds_discrete_ints[j,1:].min(), self.ds_discrete_ints[j,1:].max()] for j in range(0,len(self.ds.data))]
        self.ds_discrete_reconstructed = np.array([list(itertools.chain.from_iterable(itertools.repeat(minmax_normalize_number(self.ds_discrete_ints[j,i], stats[j][1], stats[j][0], i), diffs[i]) for i in range(0, len(diffs)))) for j in range(0,len(self.ds.data))])
        
    def copy(self):
        mycopy = Scheme(ds=self.ds, options=self.options)
        mycopy.reset()
        mycopy.cuts = self.cuts.copy()
        mycopy.error_rate = self.error_rate
        mycopy.rank = self.rank
        mycopy.crowding_distance = self.crowding_distance
        mycopy.domination_count = self.domination_count
        mycopy.dominated_set = self.dominated_set.copy()
        mycopy.ds_discrete_ints = self.ds_discrete_ints.copy()
        mycopy.ds_discrete_strings = self.ds_discrete_strings.copy()
        mycopy.ds_discrete_reconstructed = self.ds_discrete_reconstructed.copy()
        mycopy.fitness_functions = self.fitness_functions.copy()
        mycopy.confusion_matrix = self.confusion_matrix.copy()
        mycopy.surrogate_values = self.surrogate_values.copy()
        mycopy.isEvaluatedOriginal = self.isEvaluatedOriginal
        mycopy.isEvaluatedSurrogate = self.isEvaluatedSurrogate
        return mycopy
    
    def evaluate(self, model=None): #Modificar a CIAPP
        no_eval = 0
        if not model:
            self.fitness_functions.values = []
            self.discretize()
            self.confusion_matrix.create(self.ds_discrete_strings)
            self.fitness_functions.evaluate(self)  
            no_eval = 1
            self.isEvaluatedOriginal = True            
        else:
            self.surrogate_values = [np.nan] * self.fitness_functions.no_functions
            data, ff = self.to_vector()
            if len(model.models) > 3:
                print("Scheme.evaluate.len(model.models): ", len(model.models))
            surrs = []
            for i in range(0,model.no_models):
                y_pred = model.models[i].predict(data)
                surrs.append(float(y_pred))
            if len(surrs):
                print("Aquí mero es")    
            self.surrogate_values = surrs.copy()
            self.fitness_functions.values = surrs.copy()
            self.isEvaluatedSurrogate = True
            no_eval = 0
            del(data)
            del(ff)
            
        return no_eval
                
    def mutate_alphs(self, alphs):
        new_alphs = []
        for alph in alphs:
            alph_inits = []
            for interval in alph:
                alph_inits.append(interval[0])
            #print("before:",alph_inits)
            alphs_mut = set()
            alphs_mut.add(alph_inits[0])
            for i in range(1, len(alph_inits)):
                if random.random() > self.options.pm:
                    alphs_mut.add(random.uniform(self.ds.limites[0], self.ds.limites[1]))
                else:
                    alphs_mut.add(alph_inits[i])
            #print("after:", sorted(list(alphs_mut)))
            new_alphs.append(to_paired_value(sorted(list(alphs_mut.union({self.ds.limites[0], self.ds.limites[1]+1})))))
        return new_alphs
    
    def mutate(self):
        inits, _, alphs = self.extract_data()
        for j in range(1, len(inits)):
            porc = random.random()
            if (porc <= self.options.pm) and (len(inits)<self.ds.dimensions[1]):
                try:
                    new_value = random.choice([i for i in range(conf.MIN_NUMBER_OF_WORD_CUTS, self.ds.dimensions[1]) if i not in inits])
                    inits[j] = new_value
                except IndexError:
                    print("Error: please provide a name and a repeat count to the script.")
        alphs_mutate = self.mutate_alphs(alphs)
        res = {inits[i]: alphs_mutate[i] for i in range(0,len(inits))}
        new_res = collections.OrderedDict(sorted(res.items()))
        new_alphs = list(new_res.values())
        if 1 not in list(new_res.keys()):
            print("Aqui")
        new_cuts = to_paired_value(sorted(list(set(new_res.keys()).union({self.ds.dimensions[1]-1}))))
        cuts_mutate = {str(new_cuts[i]): new_alphs[i] for i in range(0,len(new_cuts))}
        self.cuts = cuts_mutate.copy()
    
    def crossover(self, parent):
        inits1, _, alphs1 = self.extract_data()
        parent1 = {inits1[i]: alphs1[i] for i in range(0,len(inits1))}
        
        inits2, _, alphs2 = parent.extract_data()
        parent2 = {inits2[i]: alphs2[i] for i in range(0,len(inits2))}

        cut1 = random.randint(1,len(parent1))
        cut2 = random.randint(1,len(parent2))
        # print(cut1, cut2)

        off1_items = collections.OrderedDict(sorted(list(parent1.items())[:cut1] + list(parent2.items())[cut2:]))
        off2_items = collections.OrderedDict(sorted(list(parent2.items())[:cut2] + list(parent1.items())[cut1:]))
        
        #if 1 not in list(off1_items.keys()):
        #    print("Aqui")
        
        #if 1 not in list(off2_items.keys()):
        #    print("Aqui")
        
        new_cuts1 = to_paired_value(sorted(list(set(off1_items.keys()).union({self.ds.dimensions[1]-1}))))
        new_alphs1 = list(off1_items.values())
        new_cuts2 = to_paired_value(sorted(list(set(off2_items.keys()).union({parent.ds.dimensions[1]-1}))))
        new_alphs2 = list(off2_items.values())
        
        ''' if len(list(off1_items.keys())) == 1:
            print("Aquí: ",list(off1_items.keys()), ", cut1: ", cut1, ", cut2: ", cut2, ", parent1: ", parent1, ", parent2: ", parent2)
            print("len(new_cuts1): ", len(new_cuts1), ", new_cuts1", new_cuts1, ", len(new_cuts2): ", len(new_cuts2), ", new_cuts2:", new_cuts2)
        
        if len(list(off2_items.keys())) == 1:
            print("Aquí: ",list(off2_items.keys()), ", cut1: ", cut1, ", cut2: ", cut2, ", parent1: ", parent1, ", parent2: ", parent2)
            print("len(new_cuts1): ", len(new_cuts1), ", new_cuts1", new_cuts1, ", len(new_cuts2): ", len(new_cuts2), ", new_cuts2:", new_cuts2) '''
        
        if (len(new_cuts1) >= conf.MIN_NUMBER_OF_WORD_CUTS) and (len(new_cuts2) >= conf.MIN_NUMBER_OF_WORD_CUTS):
            off1_cuts = {}
            off2_cuts = {}
            
            off1_cuts = {str(new_cuts1[i]): new_alphs1[i] for i in range(0,len(new_cuts1))}
            off2_cuts = {str(new_cuts2[i]): new_alphs2[i] for i in range(0,len(new_cuts2))}
            
            off1 = Scheme(ds=self.ds, cuts=off1_cuts.copy(), options=self.options)
            off2 = Scheme(ds=self.ds, cuts=off2_cuts.copy(), options=self.options)
            
            return off1, off2
        else:
            return self.crossover(parent)
    
    def dominates(self, compared):
        dominate1 = 0 
        dominate2 = 0

        flag = 0

        for i in range(0,len(self.fitness_functions.values)):
            if self.fitness_functions.values[i] < compared.fitness_functions.values[i]:
                flag = -1
            elif self.fitness_functions.values[i] > compared.fitness_functions.values[1]:
                flag = 1
            else:
                flag = 0
                
            if flag == -1:
                dominate1 = 1
            
            if flag == 1:
                dominate2 = 1

        if dominate1 == dominate2:
            dominance = False

        if dominate1 == 1:
            dominance = True #f1 dominates f2
            
        if dominate2 == 1:
            dominance = False
        #dominance = np.all(self.fitness_functions.values<=compared.fitness_functions.values) and np.any(self.fitness_functions.values<compared.fitness_functions.values)
        return dominance
    
    def matlab_format(self):
        _, ends, alphs = self.extract_data()
        list_len = [len(i) for i in list(alphs)]
        maxsize = max(list_len)+1
        mat_array = []
        for i in range(0, len(ends)):
            arr = []
            arr.append(ends[i])
            for j in range(0,len(alphs[i])):
                arr.append(alphs[i][j][0])
            diff = maxsize-len(arr)
            for k in range(0,diff):
                arr.append(float("nan"))
            mat_array.append(arr)
        return mat_array, self.fitness_functions.values, self.surrogate_values, self.isEvaluatedOriginal, self.isEvaluatedSurrogate
    
    def to_vector(self, id_model):
        #print("repr_type: ",repr_type)
        if self.options.model[id_model].train_rep == "all": #Todo el esquema como un vector
            return self._to_vector_all(normalized=False)
        elif self.options.model[id_model].train_rep == "allnorm": #Todo el esquema como un vector
            return self._to_vector_all(normalized=True)
        elif self.options.model[id_model].train_rep == "numcuts": #El esquema representado por conteos de cortes
            return self._to_vector_countcode()
        elif self.options.model[id_model].train_rep == "stats": #El esquema representado como estadísticas descriptiva
            return self._to_stats_vector()
        elif self.options.model[id_model].train_rep == "cutdits": #El esquema representado por como están distribuidos los cortes
            return self._to_cutdistr_vector()
        ''' El esquema representado como: 
                no_cortes_palabras, 
                media_cortes_palabras, 
                mediana_cortes_palabras, 
                min_cortes_palabras
                max_cortes_palabras
                std_cortes_palabras
                media_alfabetos, 
                mediana_alfabetos, 
                min_alfabetos
                max_alfabetos
                std_alfabetos'''
        
    def _to_vector_all(self, normalized=False):
        lista = []
        _, ends, alphs = self.extract_data()
        for i in range(len(ends)):
            #lista.append(ends[i])
            lista.append(minmax_normalize_number(ends[i],minimum=1, maximum=self.ds.dimensions[1],ind=-1) if normalized else ends[i])
            for j in range(1,len(alphs[i])):
                #lista.append(alphs[i][j][0])
                lista.append(minmax_normalize_number(alphs[i][j][0],minimum=self.ds.limites[0], maximum=self.ds.limites[1],ind=-1) if normalized else alphs[i][j][0])
        return lista, np.array(self.fitness_functions.values)
    
    def _to_vector_countcode(self):
        inits, _, alphs = self.extract_data()
        no_wordcuts = len(inits)
        no_alphs = [len(x)-1 for x in alphs]
        return [no_wordcuts, *no_alphs], self.fitness_functions.values
    
    def _to_stats_vector(self):
        inits, _, alphs = self.extract_data()
        no_wordcuts = len(inits)
        alphs2 = [[x[0] for x in alphs[j]] for j in range(len(alphs))]
        result = [(len(i[1:]), min(i[1:]), max(i[1:]), mean(i[1:]), pstdev(i[1:])) for i in alphs2]
        res = [item for row in result for item in row]
        return [len(inits), *res], self.fitness_functions.values

    def _to_cutdistr_vector(self):
        inits, ends, alphs = self.extract_data()
        alphs_cuts = [list(sorted(set(sum(a, [])))) for a in alphs]
        norm_alph_cuts = [minmax_normalize(a).tolist() for a in alphs_cuts]
        alphs_coded = [(len(a)-1)+(a[-1]-mean(a[1:len(a)-1])) for a in norm_alph_cuts]
        norm_time_cuts = minmax_normalize(np.array([*inits, ends[len(ends)-1]])).tolist()
        
        if len(inits) == 1:
            time_cuts_coded = len(norm_time_cuts)-1 + 0.0
        else:
            time_cuts_coded = len(norm_time_cuts)-1 + (norm_time_cuts[-1] - mean(norm_time_cuts[1:len(norm_time_cuts)-1]))
        
        #time_cuts_coded = len(norm_time_cuts)-1 + (norm_time_cuts[-1] - mean(norm_time_cuts[1:len(norm_time_cuts)-1]))
        return [time_cuts_coded, *alphs_coded], self.fitness_functions.values
    
    def classify(self, UsingTest, set_type, UsingDiscrete):
        clf = tree.DecisionTreeClassifier()
        
        train = self.ds.data
        feature_names = ["X"+str(i) for i in range(0,self.ds.dimensions[1])]
        
        if UsingTest:
            test_ds = Dataset(self.ds.id, '_TEST', False)
            test = test_ds.data
            if UsingDiscrete:
                train = self.discretize_int(train)
                test = self.discretize_int(test)
                feature_names = ["X"+str(i) for i in range(0,train.shape[1])]
                
            train_data, train_target = separate_data_target(train)
            test_data, test_target = separate_data_target(test)
            
            if set_type == "WithoutCV":
                self.classifier.ClassificationTrainTest(train_data, train_target, test_data, test_target)
            
            if set_type == "WithCV":
                self.classifier.ClassificationCV(test_data, test_target)
        
        else:
            if set_type == "original":
                original_ds = Dataset(self.ds.id, '', False)
                original = original_ds.data
                if UsingDiscrete:
                    original = self.discretize_int(original)
                data_cv = original
            if set_type == "train":
                if UsingDiscrete:
                    train = self.discretize_int(train)
                data_cv = train
            if set_type == "test":
                test_ds = Dataset(self.ds.id, '_TEST', False)
                test = test_ds.data
                if UsingDiscrete:
                    test = self.discretize_int(test)
                data_cv = test
            feature_names = ["X"+str(i) for i in range(0,data_cv.shape[1])]    
            data, target = separate_data_target(data_cv)
            self.classifier.ClassificationCV(data, target)
            
        self.DecisionTreeGraphText = tree.export_text(self.classifier.method, feature_names=feature_names)
        self.DecisionTreeGraph = tree.export_graphviz(self.classifier.method, feature_names=feature_names)
    
    def ExportGraph(self, folder, Location, Selector):
        if not Location:
            directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"+folder+"/"+self.ds.name+"/Trees"
        else:
            directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"+Location+"/"+folder+"/"+self.ds.name+"/Trees"
        
        FileName = "Arbol_best_"+Selector
        
        if not os.path.isdir(directory):
            os.mkdir(directory)
        
        text_file = open(directory+"/"+FileName+".txt", "w")
        n = text_file.write(self.DecisionTreeGraphText)
        text_file.close()
        
        graph = graphviz.Source(self.DecisionTreeGraph)
        graph.render(filename=FileName, directory=directory, format='tiff', cleanup=True)
        tikztex = dot2tex.dot2tex(self.DecisionTreeGraph, format='tikz', crop=True, output=directory+"/"+FileName+".tex")
    
    def ExportClassificationResultsByFolds(self, Location, type_selection):
        FileName = "ClassificationResultsByFolds_"+type_selection
        directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"+Location+"/MODiTS/"+self.ds.name
        if not os.path.isdir(directory):
            os.mkdir(directory)
        
        stats, headers = self.classifier.metrics2matriz()
            
        DF = pd.DataFrame(stats)
        #print(DF)
        DF.to_csv(directory+"/"+FileName+".csv", header=headers)
    
    def is_contained_in_list(self, list_schemes):
        res = False
        for sc in list_schemes:
            if sc.cuts == self.cuts:
                res = True
        return res    
        
    def prediction_measures(self, id_model):
        obs = []
        pred = []
        #print("Scheme.prediction_measure.fitness_functions.values[{id_model}]:".format(id_model=id_model), self.fitness_functions.values[id_model])
        #print("Scheme.prediction_measure.surrogate_values[{id_model}]:".format(id_model=id_model), self.surrogate_values[id_model])
        obs.append(self.fitness_functions.values[id_model])
        pred.append(self.surrogate_values[id_model])
        measure = RegressionMeasures(observed=obs, predicted=pred)
        #try:
        value = eval("measure."+self.options.evaluation_measure)()
        #except ValueError:
        #    value = 0
        return value
        
    '''public void ExportErrorRates(DataSet ds, String folder, String Location, String type_selection) {
        String FileName = "ErrorRates_"+type_selection;
        
        String directory = System.getProperty("user.dir")+"/Results"+Location+'/'+folder+'/'+ds.getName();
        
        File FileDir = new File(directory);
        if(!FileDir.exists()) FileDir.mkdirs();

        try(  PrintWriter out = new PrintWriter( directory+"/"+FileName+".csv" )  ){
            for(double d: this.csf.eval.getErrorRatesByFolds()){
               out.println(d);
            }
        } catch (FileNotFoundException ex) {
            java.util.logging.Logger.getLogger(MOScheme.class.getName()).log(Level.SEVERE, null, ex);
        }       
    }'''