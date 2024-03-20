import numpy as np
from statistics import mean
from math import sqrt
from sklearn.metrics import mean_squared_error
from dtaidistance import dtw
from dtaidistance.dtw import lb_keogh
#from Surrogate.timeseries_learn.metrics.dtw_variants import lb_keogh

'''def LB_envelope(ts, r):
    lower_bound = []
    upper_bound = []
    for i in range(len(ts)):
        lower_bound.append(min(ts[(i-r if i-r>=0 else 0):(i+r)]))
        upper_bound.append(max(ts[(i-r if i-r>=0 else 0):(i+r)]))
    #for i in range(len(ts)):
    #    min_idx = i - r
    #    max_idx = i + r + 1
    #    if min_idx < 0:
    #        min_idx = 0
    #    if max_idx > len(ts):
    #        max_idx = len(ts)
    #    lower_bound.append(np.min(ts[min_idx:max_idx]))
    #    upper_bound.append(np.max(ts[min_idx:max_idx]))
    return lower_bound, upper_bound'''

def LB_envelope(x, r):
    U = []
    L = []
    for i in range(len(x)):
        U.append(max(x[(i-r if i-r>=0 else 0):(i+r if i+r<len(x) else len(x)-1)]))
        L.append(min(x[(i-r if i-r>=0 else 0):(i+r if i+r<len(x) else len(x)-1)]))
    #U = np.array([max(x[(i-r if i-r>=0 else 0):(i+r if i+r<len(x) else len(x)-1)]) for i in range(len(x))])
    #L = np.array([min(x[(i-r if i-r>=0 else 0):(i+r if i+r<len(x) else len(x)-1)]) for i in range(len(x))])
    return np.array(L), np.array(U)

def LB_Keogh(query,candidate,r=1):
    #LB_sum=0
    #if len(s1) <= len(s2):
    #    sl = s1
    #    sg = s2
    #else: 
    #    sl = s2
    #    sg = s1
        
    min_bound = min(len(query), len(candidate))
    query = np.array(query[:min_bound])
    candidate = np.array(candidate[:min_bound])
    
    #lb_envolve
    
    L, U = LB_envelope(candidate, r)
    
    return sqrt(sum((query[query<L]-L[query<L])**2) + sum((query[query>U]-U[query>U])**2))
    
    #for i in range(len(sl)):
    #    if sl[i]>U[i]:
    #        LB_sum=LB_sum+(sl[i]-U[i])**2
    #    elif sl[i]<L[i]:
    #        LB_sum=LB_sum+(sl[i]-L[i])**2
    
    #return sqrt(LB_sum)

class KNNLBDTW:
    def __init__(self, n_neighbors = 0, is_regression=False):
        self.n_neighbors = n_neighbors
        #self.knn = None
        self.x = []
        self.y = []
        self.is_regression = is_regression
        
    def set_n_neighbors(self, n_neighbors):
        self.n_neighbors = n_neighbors
    
    def get_neighbors(self, test_row):
        distances = list()
        
        min_dist=float('inf')
        for i in range(0, len(self.x)):
            if LB_Keogh(test_row,np.array(self.x[i]),1)<min_dist:
            #window = round(max(len(test_row),len(np.array(self.x[i])))*0.1);
            #print(window)
            #if lb_keogh(test_row, np.array(self.x[i]),window=window)<min_dist:
            #if len(test_row) != len(np.array(self.x[i])):
            #    print("len(test_row): ",test_row.shape, " len(np.array(self.x[i])): ",np.array(self.x[i]).shape)
            #min_bound = min(len(test_row), len(np.array(self.x[i])))
            #if lb_keogh(test_row[:min_bound], ts_candidate=np.array(self.x[i])[:min_bound]) < min_dist:
                dist = dtw.distance_fast(test_row,  np.array(self.x[i]), use_pruning=True)
                if dist<min_dist:
                    min_dist=dist
                    #closest_seq=j
                    distances.append((self.y[i], dist))
                    
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        #print(len(distances), self.n_neighbors)
        for i in range(0,self.n_neighbors):
            #print(self.n_neighbors, i, len(distances))
            if i < len(distances):
                neighbors.append(distances[i][0])
        return neighbors
    
    
    def fit(self, X_train, Y_train):
        self.x = X_train
        self.y = Y_train
        
    def predict_set(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.predict_row(np.array(X[i])))
        return np.array(y_pred)
        
    def predict_row(self, X_test):
        neighbors = self.get_neighbors(X_test)
        if not self.is_regression:
            output_values = [row for row in neighbors]
            y_pred = max(set(output_values), key=output_values.count)
        else: 
            y_pred = mean(neighbors)
        return y_pred
        
    def classifier(self, X_test, y_test):
        y_pred = self.predict_set(X_test)
        correct = 0
        if not self.is_regression:
            for i in range(0,len(y_pred)):
                if y_pred[i] == y_test[i]:
                    correct += 1
            error_rate = 1 - (correct/len(y_pred))
        else:
            error_rate = mean_squared_error(y_test, y_pred)
        return  error_rate #misclassification error
        
    def copy(self):
        knnlbdtw = KNNLBDTW(self.n_neighbors)
        knnlbdtw.x = self.x.copy()
        knnlbdtw.y = self.y.copy()
        return knnlbdtw