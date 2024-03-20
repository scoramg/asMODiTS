import ast, os, math
import glob
from pathlib import Path
import warnings
import numpy as np

#import scipy.io
#from tslearn.metrics import dtw as dtw_ts, soft_dtw, lcss
#from dtaidistance import dtw
#from Surrogate import fastdtw as fd
#from DistanceMeasures.dtwupd import dtwupd
#from DistanceMeasures.fastdtw import fastdtw
#import asyncio

def to_paired_value(value_list):
    return [[value_list[i], value_list[i + 1]] for i in range(len(value_list)-1)]

def key_to_list(key):
    return ast.literal_eval(key)

def init_dictionary(keys):
    conteo = {}
    for i in range(0, len(keys)):
        conteo[keys[i]]=0
    return conteo

def find_last_file(path, substr=""):
    count = 0
    files = []
    last_file = None
    
    for file in os.listdir(path):
        if(file.find(substr) >= 0):
            files.append(file)
            count += 1
    files.sort(reverse=True)
    if files:
        last_file = files[0]
    return count, last_file

def find_file(path, include=None, exclude=None):
    p = Path(path)
    file_name = None
    file_version = -1
    if include and exclude:
        query = "if '{p1}' in str(fn) and '{p2}' not in str(fn)".format(p1=include,p2=exclude)
    elif include and not exclude:
        query = "if '{p1}' in str(fn)".format(p1=include)
    elif not include and exclude:
        query = "if '{p1}' in str(fn)".format(p1=exclude)
    else:
        query = ""
    files = eval("[fn for fn in p.glob('*.pkl') {query}]".format(query=query))
    count = len(files)
    if count > 0:
        f = max(files)
        file_name = Path(f).name
        name = file_name.split(".")
        name_parts = name[0].split('_')
        last_part = name_parts[len(name_parts)-1]
        file_version = last_part[1:]
    return int(file_version), file_name

def delete_files_pattern(dir, pattern):
    for item in glob.iglob(dir+pattern, recursive=True):
        try:
            os.remove(item)
        except PermissionError:
            print("No se pudo eliminar por ahora {pat}".format(pat=pattern))

def find_directory(name):
    """ funcion que busca una carpeta y devuelve la ruta de esa carpeta """
    encontro = False
    dir_path = __file__
    dir_ds = ""
    while encontro is False:
        start = os.path.dirname(os.path.realpath(dir_path))
        for dirpath, dirnames, _ in os.walk(start):
            for dirname in dirnames:
                if dirname == name:
                    encontro = True
                    dir_ds = os.path.join(dirpath, dirname)
                    break
            if encontro:
                break
        dir_path = os.path.dirname(dir_path)
    return dir_ds

def minmax_normalize(df, minimum=None, maximum=None):
    if not minimum:
        if type(df) == list:
            minimum = min(df)
            #maximum = max(df)
        else:
            minimum = df.min()
            #maximum = df.max()
    if not maximum:
        if type(df) == list:
            #minimum = min(df)
            maximum = max(df)
        else:
            #minimum = df.min()
            maximum = df.max()
    norm = 0
    if (maximum - minimum) == 0:
        norm = df/maximum
    else: 
        norm = (df - minimum) / (maximum - minimum)
    return norm

def minmax_normalize_number(number, maximum, minimum, ind=-1):
    norm = 0
    if ind == 0:
        norm = number
    elif (maximum - minimum) == 0:
        norm = number/maximum
    else: 
        norm = (number - minimum) / (maximum - minimum)
    return norm

def normalize_matrix(matrix, minimum=None, maximum=None):
    bd_norm = np.empty([len(matrix), len(matrix[0,:])], dtype=None)
    for i in range(0,len(matrix)):
        bd_norm[i,0] = matrix[i,0]
        np.copyto(bd_norm[i,1:],minmax_normalize(np.array(matrix[i,1:]),minimum=minimum, maximum=maximum))
    return bd_norm

def getNoOpponents(pop_size):
    return math.floor(pop_size * 0.1)

def compare_by_rank_crowdistance(scheme1, scheme2):
    if scheme1 is None:
        return 1
    elif scheme2 is None:
        return -1
    if scheme1.rank < scheme2.rank:
        return -1
    elif scheme1.rank > scheme2.rank:
        return 1
    elif scheme1.crowding_distance > scheme2.crowding_distance:
        return -1
    elif scheme1.crowding_distance < scheme2.crowding_distance:
        return 1
    return 0

def str_to_list(string):
    l = []
    n = len(string)
    a = string[1:n-1]
    print(type(string))
    a = a.split(',')
    for i in a:
        l.append(int(i))
    return l

def separate_data_target(data):
    target = []
    variables = []
    for i in range(0,len(data)):
        target.append(int(data[i,0]))
        variables.append(data[i,:])
    return np.array(variables), np.array(target)  

def check_dims(X, X_fit_dims=None, extend=True, check_n_features_only=False):
        """Reshapes X to a 3-dimensional array of X.shape[0] univariate
        timeseries of length X.shape[1] if X is 2-dimensional and extend
        is True. Then checks whether the provided X_fit_dims and the
        dimensions of X (except for the first one), match.

        Parameters
        ----------
        X : array-like
            The first array to be compared.
        X_fit_dims : tuple (default: None)
            The dimensions of the data generated by fit, to compare with
            the dimensions of the provided array X.
            If None, then only perform reshaping of X, if necessary.
        extend : boolean (default: True)
            Whether to reshape X, if it is 2-dimensional.
        check_n_features_only: boolean (default: False)

        Returns
        -------
        array
            Reshaped X array

        Examples
        --------
        >>> X = numpy.empty((10, 3))
        >>> check_dims(X).shape
        (10, 3, 1)
        >>> X = numpy.empty((10, 3, 1))
        >>> check_dims(X).shape
        (10, 3, 1)
        >>> X_fit_dims = (5, 3, 1)
        >>> check_dims(X, X_fit_dims).shape
        (10, 3, 1)
        >>> X_fit_dims = (5, 3, 2)
        >>> check_dims(X, X_fit_dims)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Dimensions (except first) must match! ((5, 3, 2) and (10, 3, 1)
        are passed shapes)
        >>> X_fit_dims = (5, 5, 1)
        >>> check_dims(X, X_fit_dims, check_n_features_only=True).shape
        (10, 3, 1)
        >>> X_fit_dims = (5, 5, 2)
        >>> check_dims(
        ...     X,
        ...     X_fit_dims,
        ...     check_n_features_only=True
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Number of features of the provided timeseries must match!
        (last dimension) must match the one of the fitted data!
        ((5, 5, 2) and (10, 3, 1) are passed shapes)

        Raises
        ------
        ValueError
            Will raise exception if X is None or (if X_fit_dims is provided) one
            of the dimensions of the provided data, except the first, does not
            match X_fit_dims.
        """
        if X is None:
            raise ValueError('X is equal to None!')

        if extend and len(X.shape) == 2:
            warnings.warn('2-Dimensional data passed. Assuming these are '
                        '{} 1-dimensional timeseries'.format(X.shape[0]))
            X = X.reshape((X.shape) + (1,))

        if X_fit_dims is not None:
            if check_n_features_only:
                if X_fit_dims[2] != X.shape[2]:
                    raise ValueError(
                        'Number of features of the provided timeseries'
                        '(last dimension) must match the one of the fitted data!'
                        ' ({} and {} are passed shapes)'.format(X_fit_dims,
                                                                X.shape))
            else:
                if X_fit_dims[1:] != X.shape[1:]:
                    raise ValueError(
                        'Dimensions of the provided timeseries'
                        '(except first) must match those of the fitted data!'
                        ' ({} and {} are passed shapes)'.format(X_fit_dims,
                                                                X.shape))

        return X

def delete_duplicated(data):
    result = []
    for sh in data:
        if not sh.is_contained_in_list(result):
            result.append(sh.copy())
    return result

"""def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)"""

""" def get_distance(x, test_row, metric="euclidean", window=0, useGPU=False, useNumba=False):
    dist = math.inf
    if metric == "dtw":
        #print("dtw")
        dist = dtw_ts(np.array(test_row),  np.array(x), global_constraint="sakoe_chiba", sakoe_chiba_radius=window)
    #elif metric == "cdtw":
    #    dist = dtwupd(np.array(test_row),  np.array(x), window) #Se recomienda el 10% del tamaño de la base de datos como w
    elif metric == "soft_dtw":
        #print("soft_dtw")
        dist = soft_dtw(np.array(test_row),  np.array(x), gamma=1)
    elif metric == "dtw_fast":
        #print("dtw_fast")
        dist = dtw.distance_fast(np.array(test_row),  np.array(x), use_pruning=True, window=window)
        #dist,_ = fastdtw.fastdtw(np.array(test_row),  np.array(x), numba=useNumba)
    elif metric == "lcss":
        #print("lcss")
        dist = lcss(s1=np.array(test_row),  s2=np.array(x), eps=0.05,sakoe_chiba_radius=window)
    elif metric == "euclidean":
        sum = 0
        for i in range(len(x)):
            sum += (x[i] - test_row[i]) ** 2
        dist = np.sqrt(sum)
    return dist """
    
""" def eucl_dist(x, y):
    
    #Usage
    #-----
    #L2-norm between point x and y
    #Parameters
    #----------
    #param x : numpy_array
    #param y : numpy_array
    #Returns
    #-------
    #dist : float
    #       L2-norm between x and y
    
    dist = np.linalg.norm(x - y)
    return dist

def kmeans(X, k, max_iters):
    centroids = X[np.random.choice(range(len(X)), k)]
    #centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    converged = False
    current_iter = 0
    while (not converged) and (current_iter < max_iters):
        cluster_list = [[] for i in range(len(centroids))]
        for x in X:  # Go through each data point
            distances_list = []
            for c in centroids:
                distances_list.append(get_distance(c, x))
            cluster_list[int(np.argmin(distances_list))].append(x)
        cluster_list = list((filter(None, cluster_list)))
        prev_centroids = centroids.copy()
        centroids = []
        for j in range(len(cluster_list)):
            centroids.append(np.mean(cluster_list[j], axis=0))
        pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))
        #print('K-MEANS: ', int(pattern))
        converged = (pattern == 0)
        current_iter += 1
    return np.array(centroids), [np.std(x) for x in cluster_list] """