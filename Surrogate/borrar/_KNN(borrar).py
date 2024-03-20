#https://github.com/scikit-learn/scikit-learn/blob/083ab6fcb6bfe2b4f454befcb1c585267b39e5fc/sklearn/utils/validation.py#L629
#https://tslearn.readthedocs.io/en/stable/gen_modules/neighbors/tslearn.neighbors.KNeighborsTimeSeriesRegressor.html?highlight=%20%20KNeighborsTimeSeriesRegressor
from tslearn.neighbors import KNeighborsTimeSeriesClassifier, KNeighborsTimeSeriesRegressor
import numpy as np
from statistics import mean
from sklearn.metrics import mean_squared_error

def ConvertObjectTypeTrain(X, dtype="float"):
    y = []
    for i in range(0,X.shape[0]):
        #y.append(X[i].astype(dtype))
        aux=np.array(X[i],dtype=dtype)
        y.append(aux)
    return np.asarray(y, dtype=object)

class KNNModel:
    def __init__(self, n_neighbors = 1, is_regression=False, metric="dtw"):
        self.n_neighbors = n_neighbors
        self.x = []
        self.y = []
        self.is_regression = is_regression
        self.metric = metric
        self._classifier = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbors,metric=self.metric)
        if self.is_regression:
            self._classifier = KNeighborsTimeSeriesRegressor(n_neighbors=self.n_neighbors,metric=self.metric)
    
    def set_n_neighbors(self, n_neighbors):
        self.n_neighbors = n_neighbors
    
    def fit(self, X_train, Y_train):
        #print(X_train.tolist())
        #print("----------------")
        #self.x = X_train.tolist()#ConvertObjectTypeTrain(X_train)
        #print(len(self.x))
        #print(self.x)
        #print("****************************")
        #self.y = X_train.tolist()#ConvertObjectTypeTrain(Y_train)
        #print(getattr(self.x, "dtype", None), getattr(self.x, "dtype", None))
        self.x = X_train
        self.y = Y_train
        self._classifier.fit(self.x, self.y)
    
    def predict_row(self, X_test):
        y_pred = self._classifier.predict(X_test)
        return y_pred
    
    def predict_set(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.predict_row(np.array(X[i])))
        return np.array(y_pred)
    
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
            
            