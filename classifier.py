import math
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn import tree, metrics
from sklearn.preprocessing import label_binarize

class Classifier:
    def __init__(self, clases, clf_id=0):
        self.clases = clases.copy()
        self.clases.sort()
        self.clf_id = clf_id
        if self.clf_id == 0:
            self.method = tree.DecisionTreeClassifier(criterion='entropy') #Va el algoritmo de minería de datos a usar como arboles de decision
        #self.accuracy = 0.0
        #self.error_rate = 0.0
        #self.f1_measure = 0.0
        #self.precision = 0.0
        #self.recall = 0.0
        #self.roc_auc = 0.0
        self.predictions = []
        self.accuracies = []
        self.error_rates = []
        self.f1s_measures = []
        #self.precisions = []
        #self.recalls = []
        #self.roc_aucs = []
        self.confusion_matrix = []
        self.metrics_class = {}
    
    def __del__(self):
        del(self.clf_id)
        del(self.method)
        #del(self.accuracy)
        #del(self.error_rate)
        #del(self.f1_measure)
        #del(self.precision)
        #del(self.recall)
        #del(self.roc_auc)
        del(self.predictions)
        del(self.accuracies)
        del(self.error_rates)
        del(self.f1s_measures)
        #del(self.precisions)
        #del(self.recalls)
        #del(self.roc_aucs)
        del(self.confusion_matrix)
        del(self.metrics_class)
        
    def my_binarizer(self,arr):
        arr_bin = np.zeros((len(arr),len(self.clases)))
        for i in range(0,len(arr)):
            arr_bin[i,arr[i]] = 1
        return arr_bin
        
    def ClassificationTrainTest(self, train_data, train_target, test_data, test_target):
        self.accuracies = []
        self.error_rates = []
        self.f1s_measures = []
        self.precisions = []
        #self.recalls = []
        #self.roc_aucs = []
        self.method = self.method.fit(train_data, train_target)
        self.predictions = self.method.predict(test_data)
        #self.accuracy = metrics.accuracy_score(test_target, self.predictions)
        #self.error_rate = 1 - metrics.accuracy_score(test_target, self.predictions)
        #self.f1_measure =  metrics.f1_score(test_target, self.predictions, average='weighted', zero_division=0)
        #self.precision = metrics.precision_score(test_target, self.predictions, average='weighted', zero_division=0)
        #self.recall = metrics.recall_score(test_target, self.predictions, average='weighted', zero_division=0)
        #self.roc_auc = metrics.roc_auc_score(test_target, self.predictions, average='weighted')
        #self.confusion_matrix = metrics.confusion_matrix(test_target, self.predictions)
        self.get_metrics_class(test_target)
    
    def ClassificationCV(self, data, target, cv=10):
        self.predictions = []
        self.accuracies = []
        self.error_rates = []
        self.f1s_measures = []
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True)
        for train_index, test_index in skf.split(data, target):
            #print(train_index.shape)
            #print(train_index)
            #print(type(data))
            data_train, data_test = data[train_index], data[test_index]
            target_train, target_test = target[train_index], target[test_index]
            self.method = self.method.fit(data_train, target_train)
            test_pred = self.method.predict(data_test)
            self.accuracies.append(metrics.accuracy_score(target_test, test_pred))
            self.error_rates.append(1-metrics.accuracy_score(target_test, test_pred))
            self.f1s_measures.append(metrics.f1_score(target_test, test_pred, average='weighted', zero_division=0))
            #self.precisions.append(metrics.precision_score(target_test, test_pred, average='weighted', zero_division=0))
            #self.recalls.append(metrics.recall_score(target_test, test_pred, average='weighted', zero_division=0))
            #self.roc_aucs.append(metrics.roc_auc_score(target_test, test_pred, average='weighted', multi_class='ovr'))
            #for pr in test_pred:
            #    self.predictions.append(pr)
        self.predictions = cross_val_predict(self.method, data, target, cv=skf)
        #self.accuracy = np.array(self.accuracies).mean()
        #self.error_rate = np.array(self.error_rates).mean()
        #self.f1_measure = np.array(self.f1s_measures).mean()
        #self.precision = np.array(self.precisions).mean()
        #self.recall = np.array(self.recalls).mean()
        #self.roc_aucs = np.array(self.roc_auc).mean()
        #self.confusion_matrix = metrics.confusion_matrix(target, self.predictions)
        self.get_metrics_class(target)
    
    def export_tree(self, feature_names):
        if self.clf_id ==0:
            return tree.export_text(self.method,feature_names=feature_names)
        else:   
            return ""
    
    def get_metrics_class(self, target):
        self.confusion_matrix = metrics.confusion_matrix(target, self.predictions)
        target_bin = self.my_binarizer(target)
        pred_bin = self.my_binarizer(self.predictions)
        tp_total = 0
        tn_total = 0
        fp_total = 0
        fn_total = 0
        for k in range(0,len(self.clases)):
            tp = self.TP(k)
            tn = self.TN(k)
            fp = self.FP(k)
            fn = self.FN(k)
            #tp_total += tp
            self.metrics_class[self.clases[k]] = {}
            self.metrics_class[self.clases[k]]['TP'] = tp
            self.metrics_class[self.clases[k]]['TN'] = tn
            self.metrics_class[self.clases[k]]['FP'] = fp
            self.metrics_class[self.clases[k]]['FN'] = fn
            self.metrics_class[self.clases[k]]['TPR'] = self.TPR(TP=tp, FN=fn)
            self.metrics_class[self.clases[k]]['TNR'] = self.TNR(TN=tn, FP=fp)
            self.metrics_class[self.clases[k]]['FPR'] = self.FPR(FP=fp, TN=tn)
            self.metrics_class[self.clases[k]]['FNR'] = self.FNR(FN=fn, TP=tp)
            self.metrics_class[self.clases[k]]['Recall'] = self.TPR(TP=tp, FN=fn)
            self.metrics_class[self.clases[k]]['Precision'] = self.Precision(TP=tp, FP=fp)
            self.metrics_class[self.clases[k]]['FMeasure'] = self.FMeasure(TP=tp, FP=fp, FN=fn)
            self.metrics_class[self.clases[k]]['MCC'] = self.MatthewsCorrelationCoefficient(TP=tp,TN=tn,FP=fp,FN=fn)
            self.metrics_class[self.clases[k]]['AUC'] = self.AUC(k, target_bin, pred_bin)
        
        #self.metrics_class['ACC'] = self.Accuracy(TP=tp_total)
        self.metrics_class['ACC'] = metrics.accuracy_score(target, self.predictions)    
        self.metrics_class['ErrorRate'] = 1 - self.metrics_class['ACC']
        self.metrics_class['WTPR'] = self.WeightedTPR()
        self.metrics_class['WTNR'] = self.WeightedTNR()
        self.metrics_class['WFPR'] = self.WeightedFPR()
        self.metrics_class['WFNR'] = self.WeightedFNR()
        self.metrics_class['WTPR'] = self.WeightedTPR()
        self.metrics_class['WMCC'] = self.WeightedMatthewsCorrelation()
        self.metrics_class['WRecall'] = self.WeightedTPR()
        self.metrics_class['WPrecision'] = self.WeightedPrecision()
        self.metrics_class['WFMeasure'] = self.weightedFMeasure()
        self.metrics_class['WAUC'] = self.weightedAUC(target=target_bin, pred=pred_bin)
        
        
    def TP(self, class_index):
        return self.confusion_matrix[class_index, class_index]
    
    def TN(self, class_index):
        correct = 0
        for i in range(0, len(self.clases)):
            if i != class_index:
                for j in range(0,len(self.clases)):
                    if j != class_index:
                        correct += self.confusion_matrix[i,j]
        return correct
    
    def FP(self, class_index):
        incorrect = 0
        for i in range(0, len(self.clases)):
            if i != class_index:
                for j in range(0,len(self.clases)):
                    if j == class_index:
                        incorrect += self.confusion_matrix[i,j]
        return incorrect
    
    def FN(self, class_index):
        incorrect = 0
        for i in range(0, len(self.clases)):
            if i == class_index:
                for j in range(0,len(self.clases)):
                    if j != class_index:
                        incorrect += self.confusion_matrix[i,j]
        return incorrect
    
    def Accuracy(self, TP):
        n = TP
        d = self.confusion_matrix.sum()
        print(n,d)
        if d == 0:
            return 0
        return float(n/d)
    
    def TPR(self, TP, FN): #Sensitivity o recall
        d = TP + FN
        if d == 0:
            return 0
        return float(TP/d)
    
    def TNR(self, TN, FP): #Specificity
        d = TN + FP
        if d == 0:
            return 0
        return float(TN/d)
        
    def FPR(self,FP,TN):
        d = FP + TN
        if d == 0:
            return 0
        return float(FP/d)
    
    def FNR(self,FN,TP):
        d = FN + TP
        if d == 0:
            return 0
        return float(FN/d)
    
    def Precision(self, TP, FP):
        d = TP+FP
        if d == 0:
            return 0
        #print(float(TP/d), d, TP, FP)
        return float(TP/d)
    
    def MatthewsCorrelationCoefficient(self, TP, TN, FP, FN):
        n = (TP * TN) - (FP * FN)
        d = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        if d == 0:
            d = 1
        return n/d
    
    def FMeasure(self, TP, FP, FN):
        precision = self.Precision(TP=TP, FP=FP)
        recall = self.TPR(TP=TP, FN=FN)
        if (precision + recall) == 0:
            return 0
        return float((2 * precision * recall) / (precision + recall))
    
    def AUC(self, class_index, target, pred):
        fpr, tpr, _ = metrics.roc_curve(target[:, class_index], pred[:, class_index])
        return metrics.auc(fpr, tpr)
    
    def WeightedTPR(self):
        classCounts = np.sum(self.confusion_matrix, axis=1)
        classCountSum = np.sum(self.confusion_matrix)
        truePosTotal = 0
        for i in range(0,len(self.clases)):
            #TP = self.getTP(i)
            #FN = self.getFN(i)
            TP = self.metrics_class[self.clases[i]]['TP']
            FN = self.metrics_class[self.clases[i]]['FN']
            temp = self.TPR(TP, FN)
            truePosTotal += (temp * classCounts[i])
        return truePosTotal / classCountSum
    
    def WeightedTNR(self):
        classCounts = np.sum(self.confusion_matrix, axis=1)
        classCountSum = np.sum(self.confusion_matrix)
        trueNegTotal = 0
        for i in range(0,len(self.clases)):
            #TN = self.getTN(i)
            #FP = self.getFP(i)
            TN = self.metrics_class[self.clases[i]]['TN']
            FP = self.metrics_class[self.clases[i]]['FP']
            temp = self.TNR(TN, FP)
            trueNegTotal += (temp * classCounts[i])
        return trueNegTotal / classCountSum
    
    def WeightedFPR(self):
        classCounts = np.sum(self.confusion_matrix, axis=1)
        classCountSum = np.sum(self.confusion_matrix)
        falsePosTotal = 0
        for i in range(0,len(self.clases)):
            FP = self.metrics_class[self.clases[i]]['FP']
            TN = self.metrics_class[self.clases[i]]['TN']
            temp = self.FPR(FP,TN)
            falsePosTotal += (temp * classCounts[i])
        return falsePosTotal / classCountSum
    
    def WeightedFNR(self):
        classCounts = np.sum(self.confusion_matrix, axis=1)
        classCountSum = np.sum(self.confusion_matrix)
        falseNegTotal = 0
        for i in range(0,len(self.clases)):
            FN = self.metrics_class[self.clases[i]]['FN']
            TP = self.metrics_class[self.clases[i]]['TP']
            temp = self.FNR(FN,TP)
            falseNegTotal += (temp * classCounts[i])
        return falseNegTotal / classCountSum
    
    def WeightedMatthewsCorrelation(self):
        classCounts = np.sum(self.confusion_matrix, axis=1)
        classCountSum = np.sum(self.confusion_matrix)
        mccTotal = 0
        for i in range(0,len(self.clases)):
            TP = self.metrics_class[self.clases[i]]['TP']
            TN = self.metrics_class[self.clases[i]]['TN']
            FP = self.metrics_class[self.clases[i]]['FP']
            FN = self.metrics_class[self.clases[i]]['FN']
            temp = self.MatthewsCorrelationCoefficient(TP=TP,TN=TN,FP=FP,FN=FN)
            mccTotal += (temp * classCounts[i])
        return mccTotal / classCountSum
    
    def WeightedPrecision(self):
        classCounts = np.sum(self.confusion_matrix, axis=1)
        classCountSum = np.sum(self.confusion_matrix)
        precisionTotal = 0
        for i in range(0,len(self.clases)):
            TP = self.metrics_class[self.clases[i]]['TP']
            FP = self.metrics_class[self.clases[i]]['FP']
            temp = self.Precision(TP=TP, FP=FP)
            precisionTotal += (temp * classCounts[i])
        return precisionTotal / classCountSum
    
    def weightedFMeasure(self):
        classCounts = np.sum(self.confusion_matrix, axis=1)
        classCountSum = np.sum(self.confusion_matrix)
        FMeasureTotal = 0
        for i in range(0,len(self.clases)):
            TP = self.metrics_class[self.clases[i]]['TP']
            FP = self.metrics_class[self.clases[i]]['FP']
            FN = self.metrics_class[self.clases[i]]['FN']
            temp = self.FMeasure(TP=TP, FP=FP, FN=FN)
            FMeasureTotal += (temp * classCounts[i])
        return FMeasureTotal / classCountSum
    
    def weightedAUC(self, target, pred):
        classCounts = np.sum(self.confusion_matrix, axis=1)
        classCountSum = np.sum(self.confusion_matrix)
        aucTotal = 0
        for i in range(0,len(self.clases)):
            temp = self.AUC(i, target, pred)
            aucTotal += (temp * classCounts[i])
        return aucTotal / classCountSum
    
    def metrics2matriz(self):
        matriz = np.array([self.accuracies, self.error_rates, self.f1s_measures])
        headers = ['Acuracy', 'Error Rate', 'F1 measure'];
        return matriz.transpose(), headers
            
    ''' def ClassificationCV(self, data, target, cv=10):
        parameters = {}
        results_acc = GridSearchCV(self.method, parameters, cv=10, scoring='accuracy').fit(data, target).cv_results_
        self.accuracies = [float(value) for key,value in results_acc.items() if key.startswith("split")]
        self.accuracy = results_acc['mean_test_score']
        
        self.error_rates = [1-float(value) for key,value in results_acc.items() if key.startswith("split")]
        self.error_rate = 1 - self.accuracy
        
        results_f1 = GridSearchCV(self.method, parameters, cv=10, scoring='f1_weighted').fit(data, target).cv_results_
        self.f1s_weighted = [float(value) for key,value in results_f1.items() if key.startswith("split")]
        self.f1_weighted = results_f1['mean_test_score']
        
        results_precision = GridSearchCV(self.method, parameters, cv=10, scoring='precision_weighted').fit(data, target).cv_results_
        self.precisions = [float(value) for key,value in results_precision.items() if key.startswith("split")]
        self.precision = results_precision['mean_test_score']
        
        results_recall = GridSearchCV(self.method, parameters, cv=10, scoring='recall_weighted').fit(data, target).cv_results_
        self.recalls = [float(value) for key,value in results_recall.items() if key.startswith("split")]
        self.recall = results_recall['mean_test_score']
        
        results_roc = GridSearchCV(self.method, parameters, cv=10, scoring='roc_auc_ovo_weighted').fit(data, target).cv_results_
        self.roc_aucs = [float(value) for key,value in results_roc.items() if key.startswith("split")]
        self.roc_auc = results_roc['mean_test_score']
        
        self.predictions = cross_val_predict(dtree, X, y, cv=cv)
        
        #self.precision = np.array(cross_val_score(self.method, data, target, cv=cv, scoring='precision')).mean()
        #self.recall = np.array(cross_val_score(self.method, data, target, cv=cv, scoring='recall')).mean()
        #self.roc_auc = np.array(cross_val_score(self.method, data, target, cv=cv, scoring='roc_auc')).mean()'''
        
#from weka.classifiers import Classifier
#c = Classifier(classname='weka.classifiers.lazy.IBk', options=["-K", "1"])
#c.train('/Users/scoramg/Dropbox/Escolaridad/Doctorado en Inteligencia Artificial (IIIA UV)/Repositorios/iris.arff')
#predictions = c.predict('query.arff')