import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

class RegressionMeasures:
    
    @staticmethod
    def init_measures_values():
        return  {'MSE':np.NZERO,'R':np.NZERO,'R2':np.NZERO,'RMSE':np.NZERO,'MD':np.NZERO,'MAPE':np.NZERO,'MAE':np.NZERO}
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.values = RegressionMeasures.init_measures_values()
        #properties: observed, predicted
        if not hasattr(self,"observed"):
            self.observed = np.array([],dtype=float)
        if not hasattr(self,"predicted"):
            self.predicted = np.array([],dtype=float)
            
        # if hasattr(self,"observed") is not None:
        #     self.observed = np.array(observed.copy(),dtype=float)
        # else:
        #     self.observed = np.array([],dtype=float)
        # if predicted is not None:
        #     self.predicted = np.array(predicted.copy(),dtype=float)
        # else:
        #     self.predicted= np.array([],dtype=float)
        
    def setObserved(self, observed):
        self.observed = np.array(observed.copy(),dtype=float)
        
    def addObservedValue(self, value):
        self.observed = np.append(self.observed, value)
    
    def setPredicted(self, predicted):
        self.predicted = np.array(predicted.copy(),dtype=float)
        
    def addPredictedValue(self, value):
        self.predicted = np.append(self.predicted, value)
        
    def _clear_data(self):
        indexes = np.where(np.isnan(self.predicted))[0]
        mask = np.ones(len(self.predicted), dtype=bool)
        mask[indexes] = False
        if len(mask[indexes]) > 0:
            try:
                self.predicted = np.array(self.predicted[mask], dtype=float)
                self.observed = np.array(self.observed[mask], dtype=float)
            except TypeError:
                print("regression_measures._clear_data.Error: only integer scalar arrays can be converted to a scalar index")
                print("observed array shape:",self.observed.shape,", predicted array shape:", self.predicted.shape, ", mask array shape:",mask.shape)
                if self.observed.shape[0] > 1 and self.observed.shape[1] > 1:
                    print("observed array:",self.observed)
                if self.predicted.shape[0] > 1 and self.predicted.shape[1] > 1:
                    print("predicted array:", self.predicted)
                if mask.shape[0] > 1 and mask.shape[1] > 1:
                    print("mask array:",mask)
                assert("Exit...")
        
    @property    
    def _is_empty_obs(self):
        isEmpty = True
        if len(self.observed) > 0:
            isEmpty = False
        return isEmpty

    @property    
    def _is_empty_pred(self):
        isEmpty = True
        if len(self.predicted) > 0:
            isEmpty = False
        return isEmpty
    
    def MSE(self):
        #print("regression_measures.MSE.observed:",self.observed)
        #print("regression_measures.MSE.predicted:",self.predicted)
        self._clear_data()
        if self._is_empty_obs or self._is_empty_pred:
            self.values['MSE'] = np.NZERO
        else:    
            self.values['MSE'] = round(mean_squared_error(self.observed, self.predicted),4)
        return self.values['MSE']
    
    def MAE(self):
        self._clear_data()
        if self._is_empty_obs or self._is_empty_pred:
            self.values['MAE'] = np.NZERO
        else:
            self.values['MAE'] = round(mean_absolute_error(self.observed, self.predicted),4)
        return self.values['MAE']
    
    def RMSE(self):
        self._clear_data()
        if self._is_empty_obs or self._is_empty_pred:
            self.values['RMSE'] = np.NZERO
        else:
            self.values['RMSE'] = round(math.sqrt(self.MSE()),4)
        return self.values['RMSE']
    
    def R(self):
        if self._is_empty_obs or self._is_empty_pred:
            self.values['R'] = np.NZERO
        else:
            ff_orig = pd.Series(data=self.observed, dtype=float)
            ff_surr = pd.Series(data=self.predicted, dtype=float)
            self.values['R'] = round(ff_orig.corr(ff_surr),4)
        return self.values['R']
        
    def R2(self):
        observed_mean = np.mean(self.observed)
        sum_num = 0
        sum_den = 0
        for i in range(0,len(self.observed)):
            num = (self.predicted[i]-self.observed[i])**2
            sum_num += num
            den = (self.observed[i]-observed_mean)**2
            sum_den += den
        try:
            self.values['R2'] = round(1 - (sum_num/sum_den),4)
        except ZeroDivisionError:
            self.values['R2'] = np.NZERO
            #print("R2.len(self.observed): ",len(self.observed))
            #exit(1)
        return self.values['R2']
        
    def MD(self, j=1): # modified index of agreement (0-1), 0 -> A value of 1 indicates a perfect match, and 0 indicates no agreement at all
        observed_mean = np.mean(self.observed)
        sum_num = 0
        sum_den = 0
        for i in range(0,len(self.observed)):
            num = abs(self.observed[i]-self.predicted[i])**j
            sum_num += num
            den = abs(self.predicted[i] - observed_mean) + (abs(self.observed[i] - observed_mean)**j)
            sum_den += den
        try:
            #print("regression_measures.MD.sum_num:",sum_num)
            #print("regression_measures.MD.sum_den:",sum_den)
            self.values['MD'] = round(1 - (sum_num/sum_den),4)
        except ZeroDivisionError:
            self.values['MD'] = np.NZERO
            #print("MD.len(self.observed): ",len(self.observed))
            #print("MD.len(self.predicted): ",len(self.predicted))
            #exit(1)
        return self.values['MD']
    
    def MAPE(self):
        m=0
        if self._is_empty_obs or self._is_empty_pred:
            self.values['MAPE'] = np.NZERO
        else:
            for i in range(0,len(self.observed)):
                m += np.abs((self.observed[i] - self.predicted[i])/self.observed[i])
            self.values['MAPE'] = m/self.observed.shape[0]
        return self.values['MAPE']
    
    def compute(self):
        self.MSE()
        self.R()
        self.R2()
        self.RMSE()
        self.MD()
        self.MAPE()
        self.MAE()
    
    def export_matlab(self):
        data = {}
        data['observed'] = self.observed
        data['predicted'] = self.predicted
        for key, value in self.values.items():
            data[key] = value
        return data
        