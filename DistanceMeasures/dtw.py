import time, os, codecs
import numpy as np
from math import ceil
from dtaidistance import dtw as dtw_distance
from dtaidistance.dtw_barycenter import dba
import itertools

from .utils import cdist_generic

def LB_Keogh(self,s1,s2,r):
    '''
    Calculates LB_Keough lower bound to dynamic time warping. Linear
    complexity compared to quadratic complexity of dtw.
    '''
    LB_sum=0
    for ind,i in enumerate(s1):
        
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    
    return np.sqrt(LB_sum)

class DTW:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    @property
    def name(self):
        return self.__class__.__name__
         
    def distance(self, dataset1, dataset2, inv = False):
        if dataset2 is None:
            dataset2 = np.copy(dataset1)
            
        if inv:
            dataset1 = np.array(dataset1).T
            dataset2 = np.array(dataset2).T
            
        n1 = dataset1.shape[0]
        n2 = dataset2.shape[0]
        dist = np.empty((n1, n2), dtype=np.float64)
        
        for i,j in list(itertools.product(np.arange(0,n1),np.arange(0,n2))):
            dist[i][j] = self._distance(np.array(dataset1[i],dtype=float), np.array(dataset2[j],dtype=float))
            if np.isinf(dist[i][j]):
                print("dtw.distance.dataset1[i]:", dataset1[i])
                print("dtw.distance.dataset2[j]:", dataset2[j])
        
        return dist
    
    def _distance(self, serie1, serie2):
        serie1 = serie1[~np.isnan(serie1)]
        serie2 = serie2[~np.isnan(serie2)]
        w = int(min(len(serie1), len(serie2)) * self.options.dtw_sakoechiba_w)
        dist = dtw_distance.distance_fast(s1=serie1,s2=serie2,window=w,use_pruning=True)
        return dist
    
    def _check_sakoe_chiba_params(self, n_timestamps_1, n_timestamps_2):
            """Check and set some parameters of the sakoe-chiba band."""
            if not isinstance(n_timestamps_1, (int, np.integer)):
                raise TypeError("'n_timestamps_1' must be an integer.")
            else:
                if not n_timestamps_1 >= 2:
                    raise ValueError("'n_timestamps_1' must be an integer greater than"
                                    " or equal to 2.")
            window_size = self.options.dtw_sakoechiba_w
            if not isinstance(window_size, (int, np.integer, float, np.floating)):
                raise TypeError("'window_size' must be an integer or a float.")
            n_timestamps = max(n_timestamps_1, n_timestamps_2)

            if isinstance(window_size, (float, np.floating)):
                if not 0. <= window_size <= 1.:
                    raise ValueError("The given 'window_size' is a float, "
                                    "it must be between "
                                    "0. and 1. To set the size of the sakoe-chiba "
                                    "manually, 'window_size' must be an integer.")
                window_size = ceil(window_size * (n_timestamps - 1))
            else:
                if not 0 <= window_size <= (n_timestamps - 1):
                    raise ValueError(
                        "The given 'window_size' is an integer, it must "
                        "be greater "
                        "than or equal to 0 and lower than max('n_timestamps_1', "
                        "'n_timestamps_2')."
                    )
            
            scale = (n_timestamps_2 - 1) / (n_timestamps_1 - 1)

            if n_timestamps_2 > n_timestamps_1:
                window_size = max(window_size, scale / 2)
                horizontal_shift = 0
                vertical_shift = window_size
            elif n_timestamps_1 > n_timestamps_2:
                window_size = max(window_size, 0.5 / scale)
                horizontal_shift = window_size
                vertical_shift = 0
            else:
                horizontal_shift = 0
                vertical_shift = window_size
            return scale, horizontal_shift, vertical_shift
        
    def path(self, s1, s2):
        return dtw_distance.warping_path_fast(from_s=s1, to_s=s2)
    
    def _sakoe_chiba_band(self, n_timestamps_1, n_timestamps_2=None):
        """Compute the Sakoe-Chiba band.

        Parameters
        ----------
        n_timestamps_1 : int
            The size of the first time series.

        n_timestamps_2 : int (optional, default None)
            The size of the second time series. If None, set to `n_timestamps_1`.


        Returns
        -------
        region : array, shape = (2, n_timestamps_1)
            Constraint region. The first row consists of the starting indices
            (included) and the second row consists of the ending indices (excluded)
            of the valid rows for each column.

        References
        ----------
        .. [1] H. Sakoe and S. Chiba, “Dynamic programming algorithm optimization
            for spoken word recognition”. IEEE Transactions on Acoustics,
            Speech, and Signal Processing, 26(1), 43-49 (1978).

        Examples
        --------
        >>> from pyts.metrics import sakoe_chiba_band
        >>> print(sakoe_chiba_band(5, window_size=0.5))
        [[0 0 0 1 2]
        [3 4 5 5 5]]

        """
        if n_timestamps_2 is None:
            n_timestamps_2 = n_timestamps_1
        scale, horizontal_shift, vertical_shift = self._check_sakoe_chiba_params(n_timestamps_1, n_timestamps_2)

        lower_bound = scale * (np.arange(n_timestamps_1) - horizontal_shift) - vertical_shift
        lower_bound = np.round(lower_bound, 2)
        lower_bound = np.ceil(lower_bound)
        upper_bound = scale * (np.arange(n_timestamps_1) + horizontal_shift) + vertical_shift
        upper_bound = np.round(upper_bound, 2)
        upper_bound = np.floor(upper_bound) + 1
        region = np.asarray([lower_bound, upper_bound]).astype('int64')
        region = (region, n_timestamps_1, n_timestamps_2)
        return region
    
    def cdist(self, dataset1, dataset2, n_jobs=None, verbose=0):
        return cdist_generic(dist_fun=self.distance, dataset1=dataset1, dataset2=dataset2,
                        n_jobs=n_jobs, verbose=verbose,
                        compute_diagonal=False)
    
    def barycenter_averaging(self, X, init_barycenter=None):
        w = int(min(len(X), len(init_barycenter)) * self.options.dtw_sakoechiba_w)
        print("dtw.barycenter_aveeraging.init_barycenter:", np.array(list(init_barycenter.ravel())))
        return dba(s=X.ravel(), c=np.array(list(init_barycenter.ravel())), use_c=True, window=w, use_pruning=True)