import numpy as np
from statistics import mean
from sklearn.metrics import mean_squared_error
#from tslearn.metrics import dtw as dtw_ts, soft_dtw
#from dtaidistance import dtw
from math import sqrt, inf

from Utils.utils import get_distance

def LB_envelope(x, r):
    U = []
    L = []
    for i in range(len(x)):
        U.append(max(x[(i-r if i-r>=0 else 0):(i+r if i+r<len(x) else len(x)-1)]))
        L.append(min(x[(i-r if i-r>=0 else 0):(i+r if i+r<len(x) else len(x)-1)]))
    #U = np.array([max(x[(i-r if i-r>=0 else 0):(i+r if i+r<len(x) else len(x)-1)]) for i in range(len(x))])
    #L = np.array([min(x[(i-r if i-r>=0 else 0):(i+r if i+r<len(x) else len(x)-1)]) for i in range(len(x))])
    return np.array(L), np.array(U)

def LB_Kim(query, candidate):
    smallest = abs(min(query)-min(candidate))
    greatest = abs(max(query)-max(candidate))
    firsts = abs(query[0]-candidate[0])
    lasts = abs(query[-1]-candidate[-1])
    return max(firsts,lasts,greatest,smallest)
    

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

def LB_Keogh_unequal(query,candidate,r=1):
    #LB_sum=0
    #if len(s1) <= len(s2):
    #    sl = s1
    #    sg = s2
    #else: 
    #    sl = s2
    #    sg = s1
    
    min_bound = min(len(query), len(candidate))
    s1 = np.array([*query, *candidate[min_bound:len(candidate)]])
    s2 = np.array([*candidate, *query[min_bound:len(query)]])
    query = s1
    candidate = s2
    
    #lb_envolve
    
    L, U = LB_envelope(candidate, r)
    
    return sqrt(sum((query[query<L]-L[query<L])**2) + sum((query[query>U]-U[query>U])**2))

def num_word_segments_by_array(x):
    word_segments = [val for val in x if isinstance(val, (int))]
    return len(word_segments)

def CalcDist(x,y, window):
    #print("Word cut - x:", num_word_segments_by_array(x), ", Word cut - y", num_word_segments_by_array(y), ", dif:",abs(num_word_segments_by_array(x) - num_word_segments_by_array(y)), ", window:", window)
    if abs(num_word_segments_by_array(x) - num_word_segments_by_array(y)) <= window:
        return True
    else: 
        return False
    
class KNNDTW:
    def __init__(self, n_neighbors = 0, is_regression=False, metric="dtw", optimizer="None", window=10):
        self.n_neighbors = n_neighbors
        #self.knn = None
        self.x = []
        self.y = []
        self.is_regression = is_regression
        self.metric = metric
        self.window = window
        self.optimizer = optimizer
        
    def set_n_neighbors(self, n_neighbors):
        self.n_neighbors = n_neighbors
        
    """ def get_distance(self, x, test_row):
        dist = inf
        if self.metric == "dtw":
            #print("dtw")
            dist = dtw_ts(np.array(test_row),  np.array(x), global_constraint="sakoe_chiba", sakoe_chiba_radius=self.window)
        elif self.metric == "soft_dtw":
            #print("soft_dtw")
            dist = soft_dtw(np.array(test_row),  np.array(x), gamma=1)
        elif self.metric == "dtw_fast":
            #print("dtw_fast")
            dist = dtw.distance_fast(np.array(test_row),  np.array(x), use_pruning=True)
        return dist """
    
    def get_neighbors(self, test_row):        
        distances = list()
        min_dist=float('inf')
        for i in range(0, len(self.x)):
            
            if self.optimizer == "None":
                #print("None")
                #dist = self.get_distance(self.x[i], test_row)
                dist = get_distance(self.x[i], test_row, self.metric, self.window)
                distances.append((self.y[i], dist))
                
            elif self.optimizer == "lb_keogh":
                if LB_Keogh_unequal(test_row,np.array(self.x[i]),1)<min_dist:
                    #print("lb_keogh")
                    #dist = self.get_distance(self.x[i], test_row)
                    dist = get_distance(self.x[i], test_row, self.metric, self.window)
                    if dist<min_dist:
                        min_dist=dist
                        #closest_seq=j
                    distances.append((self.y[i], dist))
                else:
                    distances.append((self.y[i], float('inf')))
            elif self.optimizer == "lb_kim":
                if LB_Kim(test_row,np.array(self.x[i]))<min_dist:
                    #print("lb_keogh")
                    #dist = self.get_distance(self.x[i], test_row)
                    dist = get_distance(self.x[i], test_row, self.metric, self.window)
                    if dist<min_dist:
                        min_dist=dist
                        #closest_seq=j
                    distances.append((self.y[i], dist))
                else:
                    distances.append((self.y[i], float('inf')))
            elif self.optimizer == "by_words":
                #print("A")
                if CalcDist(x=self.x[i],y=test_row, window=self.window):
                    #print("B")
                    #dist = self.get_distance(self.x[i], test_row)
                    dist = get_distance(self.x[i], test_row, self.metric, self.window)
                    distances.append((self.y[i], dist))
                else:
                    #dist = float('inf')
                    #print("C")
                    distances.append((1000000, float('inf')))
                #print("gak")
        #print(distances)
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        #print(len(distances))
        #print(self.n_neighbors)
        for i in range(self.n_neighbors):
            neighbors.append(distances[i][0])
        #print(neighbors)
        return neighbors
    
    
    def fit(self, X_train, Y_train):
        self.x = X_train
        self.y = Y_train
        
    def predict_set(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            #y_pred.append(self.predict_row(np.array(X[i])))
            y_pred.append(self.predict_row(X[i]))
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
        knndtw = KNNDTW(n_neighbors=self.n_neighbors, is_regression=self.is_regression, 
                        metric=self.metric, optimizer=self.optimizer, window=self.window)
        knndtw.x = self.x.copy()
        knndtw.y = self.y.copy()
        return knndtw