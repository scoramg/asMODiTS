import numpy as np
from numba import prange

class CountingDistance:
    def __init__(self, options):
        self.options = options

    def distance(self, dataset1, dataset2): #dataset2 conjunto de entrenamiento
        if dataset2 is None:
            dataset2 = np.copy(dataset1)
        n1 = dataset1.shape[0]
        n2 = dataset2.shape[0]
        dist = np.empty((n1, n2), dtype=np.float64)

        for i in prange(n1):
            for j in prange(n2):
                dist[i][j] = self._distance(dataset1[i], dataset2[j])
        return dist
    
    """ def decode(self, x):
        alphs = {}
        if self.options.train_rep == 0:
            key = None
            for i in range(0,len(x)):
                if isinstance(x[i], (int)):
                    key = x[i]
                    alphs[key] = []
                else :
                    alphs[key].append(x[i])
        return alphs  """             
    
    def _num_word_segments_by_array(self,x):
        word_segments = [val for val in x if isinstance(val, (int))]
        return len(word_segments)
    
    """ def to_cutdistr_vector(x):
        inits, ends, alphs = self.extract_data()
        alphs_cuts = [list(sorted(set(sum(a, [])))) for a in alphs]
        norm_alph_cuts = [minmax_normalize(a).tolist() for a in alphs_cuts]
        alphs_coded = [(len(a)-1)+(a[-1]-mean(a[1:len(a)-1])) for a in norm_alph_cuts]
        norm_time_cuts = minmax_normalize(np.array([*inits, ends[len(ends)-1]])).tolist()
        if len(inits) == 1:
            time_cuts_coded = len(norm_time_cuts)-1 + 0.0
        else:
            time_cuts_coded = len(norm_time_cuts)-1 + (norm_time_cuts[-1] - mean(norm_time_cuts[1:len(norm_time_cuts)-1])) """
    
    def _distance(self, x,y, window=1): 
        if self.options.train_rep == 0:
            dist = abs(self._num_word_segments_by_array(x) - self._num_word_segments_by_array(y)) 
        return dist