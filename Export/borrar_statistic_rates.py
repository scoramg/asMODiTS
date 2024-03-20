#from sklearn import

class StatisticRates:
    def __init__(LabelClass="", ClassificationRate=0.0, MisclassificationRate=0.0, TruePositiveRate=0.0, 
            TrueNegativeRate=0.0, Precision=0.0, Recall=0.0, F_measure=0.0, MatthewsCorrelationCoefficient=0.0, 
            AreaUnderROC=0.0, AreaUnderPRC=0.0, Accuracy=0.0, Sensitivity=0.0, Specificity=0.0):
        self.LabelClass = LabelClass
        self.ClassificationRate = ClassificationRate
        self.MisclassificationRate = MisclassificationRate
        self.TruePositiveRate = TruePositiveRate
        self.TrueNegativeRate = TrueNegativeRate
        self.Precision = Precision
        self.Recall = Recall
        self.f_measure = F_measure
        self.MatthewsCorrelationCoefficient = MatthewsCorrelationCoefficient
        self.AreaUnderROC = AreaUnderROC
        self.AreaUnderPRC = AreaUnderPRC
        self.Accuracy = Accuracy
        self.Sensitivity = Sensitivity
        self.Specificity = Specificity
        
    def __del__(self):
        del(self.LabelClass)
        del(self.ClassificationRate)
        del(self.MisclassificationRate)
        del(self.TruePositiveRate)
        del(self.TrueNegativeRate)
        del(self.Precision)
        del(self.Recall)
        del(self.f_measure)
        del(self.MatthewsCorrelationCoefficient)
        del(self.AreaUnderROC)
        del(self.AreaUnderPRC)
        del(self.Accuracy)
        del(self.Sensitivity)
        del(self.Specificity)
        
    def __str__(self):
         return self.LabelClass+","+self.ClassificationRate+","+self.MisclassificationRate+","+self.TruePositiveRate+","+ \
                self.TrueNegativeRate+","+self.Precision+","+self.Recall+","+self.f_measure+","+self.MatthewsCorrelationCoefficient+","+ \
                self.AreaUnderROC+","+self.AreaUnderPRC+","+self.Accuracy+","+self.Sensitivity+","+self.Specificity
        
    def copy(self):
        my_copy = StatisticRates(LabelClass=self.LabelClass, ClassificationRate=self.ClassificationRate, 
                                 MisclassificationRate=self.MisclassificationRate, TruePositiveRate=self.TruePositiveRate, 
                                 TrueNegativeRate=self.TrueNegativeRate, Precision=self.Precision, Recall=self.Recall,
                                 F_measure=self.F_measure, MatthewsCorrelationCoefficient=self.MatthewsCorrelationCoefficient, 
                                 AreaUnderROC=self.AreaUnderROC, AreaUnderPRC=self.AreaUnderPRC, Accuracy=self.Accuracy, 
                                 Sensitivity=self.Sensitivity, Specificity=self.Specificity)
        return my_copy
    
class StatisticRatesCollection():
    def __init__(self):
        RatesByClass = []
    
    def __del__(self):
        pass