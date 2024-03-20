from global_align import tga_dissimilarity
from sklearn.utils import check_random_state
from Utils.timeseries import to_time_series_dataset, check_equal_size, time_series_to_lists
from scipy.spatial.distance import pdist
import numpy as np
import itertools

class TGA:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.dataset1 = None
        self.dataset2 = None
        #self.options = options
    
    @property
    def sigma(self):
        random_state = None
        n_samples=100
        random_state = check_random_state(random_state)
        dataset = to_time_series_dataset(dataset=self.dataset1)
        n_ts, sz, d = dataset.shape
        if check_equal_size(dataset):
            sz = np.min([ts.size for ts in dataset])
        if n_ts * sz < n_samples:
            replace = True
        else:
            replace = False
        sample_indices = random_state.choice(n_ts * sz, size=n_samples, replace=replace)
        dists = pdist(dataset[:, :sz, :].reshape((-1, d))[sample_indices], metric="euclidean")
        if (np.median(dists) * np.sqrt(sz)) == 0:
            print("[n_ts, sz, d]: ",dataset.shape,", dists: ", dists, ", np.median(dists): ", np.median(dists))
            return 1.0
        else:
            return np.median(dists) * np.sqrt(sz)
    
    @property
    def triangular(self):
        #lenghts = [len(row) for row in self.dataset1]
        #return int(0.5 * np.median(lenghts))
        return self.options.dtw_sakoechiba_w
    
    def distance(self, dataset1, dataset2=None):
        self.dataset1 = np.copy(dataset1)
        if dataset2 is None:
            self.dataset2 = np.copy(dataset1)
        else:
            self.dataset2 = np.copy(dataset2)
        #print(self.dataset1)
        #print(self.dataset2)
        n1 = self.dataset1.shape[0]
        n2 = self.dataset2.shape[0]
        dist = np.empty((n1, n2), dtype=np.float64)
        sigma = self.sigma
        #print(sigma)
        triangular = self.triangular
        #print(triangular)
        
        for i,j in list(itertools.product(np.arange(0,n1),np.arange(0,n2))):
            s1 = np.array(self.dataset1[i])
            s1 = np.expand_dims(s1, axis=1)
            s2 = np.array(self.dataset2[j])
            s2 = np.expand_dims(s2, axis=1)
            dist[i][j] = tga_dissimilarity(s1, s2, sigma=sigma, triangular=triangular)
        return dist
        #print(self.dataset1)
        #print(self.dataset1.tolist(), type(self.dataset1))
        #s1 = np.array(time_series_to_lists(self.dataset1),ndmin=2).astype('float')
        #print(s1)
        #s2 = np.array(time_series_to_lists(self.dataset2),ndmin=2)
        #return tga_dissimilarity(s1, s2, sigma=sigma, triangular=triangular)