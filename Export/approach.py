from scipy import stats
#from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.sandbox.stats.runs import mcnemar
from Datasets.dataset import Dataset

class Approach:
    def __init__(self, ErrorRate=0, pvalue=0, StandardDeviation=0, StatisticalData=[],rank=0, 
                 name="",h0="",IsApproachBase=False,correctPredictions=[],statistic_rates=[], 
                 IsNormalDistribution=False):
        self.ErrorRate = ErrorRate
        self.pvalue = pvalue
        self.StandardDeviation = StandardDeviation
        self.StatisticalData = StatisticalData
        self.rank = rank
        self.name = name
        self.h0 = h0
        self.IsApproachBase = IsApproachBase
        self.correctPredictions = correctPredictions
        self.statistic_rates = statistic_rates #private StatisticRatesCollection statistic_rates;
        self.IsNormalDistribution = IsNormalDistribution
        
    def __del__(self):
        del(self.ErrorRate)
        del(self.pvalue)
        del(self.StandardDeviation)
        del(self.StatisticalData)
        del(self.rank)
        del(self.name)
        del(self.h0)
        del(self.IsApproachBase)
        del(self.correctPredictions)
        del(self.statistic_rates)
        del(self.IsNormalDistribution)
        
    def getIsNormalDistribution(self):
        shapiro_test = stats.shapiro(self.StatisticalData)
        self.IsNormalDistribution = shapiro_test.pvalue <= 0.05
        
    def CalculateStats(self, base, test, StatisticalTypeData):
        if test == "wilcoxon":
            res = stats.wilcoxon(base.StatisticalData, self.StatisticalData)
            #res = stats.ranksums(base.StatisticalData, self.StatisticalData)
            self.pvalue = res.pvalue
            
        if test == "mcnemar":
            res = mcnemar(base.StatisticalData, self.StatisticalData)
            self.pvalue = res.pvalue
            
        if test == "t-test":
            res = stats.ttest_ind(base.StatisticalData, self.StatisticalData)
            self.pvalue = res.pvalue
        
        if self.pvalue < 0.05:
            if StatisticalTypeData==1:
                if base.ErrorRate < self.ErrorRate:
                    self.h0 = "+"
                else:
                    self.h0 = "-"
            elif base.ErrorRate > self.ErrorRate:
                self.h0 = "+"
            else:
                self.h0 = "-"
        else:
            self.h0 = "="
            
    def copy(self):
        mycopy = Approach(ErrorRate=self.ErrorRate, pvalue=self.pvalue, StandardDeviation=self.StandardDeviation,
                          StatisticalData=self.StatisticalData,rank=self.rank,name=self.name,h0=self.h0,
                          IsApproachBase=self.IsApproachBase,correctPredictions=self.correctPredictions,
                          statistic_rates=self.statistic_rates,IsNormalDistribution=self.IsNormalDistribution)
        return mycopy

class ApproachCollection:
    def __init__(self, ds):
        self.ds = ds
        self.approaches = []
        
    def __del__(self):
        del(self.ds)
        del(self.approaches)
    
    def getApproachBase(self):
        appbase = Approach()
        for app in self.approaches:
            if app.IsApproachBase:
                appbase = app.copy()
        return appbase
    
    def getApproachesName(self):
        names = []
        for i in range(len(self.approaches)):
            names.append(self.approaches[i].name);
        return names
    
    def setRanks(self):
        mediasSet = {}
        for i in range(len(self.approaches)):
            mediasSet.add(self.approaches[i].ErrorRate)
        medias = list(sorted(mediasSet))
        for j in range(len(self.approaches)):
            rank = medias.index(self.approaches[j])+1
            self.approaches[j].rank = rank
    
    def CalculateStats(self, test, StatisticalTypeData):
        appbase = self.getApproachBase()
        for i in range(len(self.approaches)):
            self.approaches[i].CalculateStats(appbase, test, StatisticalTypeData)
    
    def PrintStatisticRates(self):
        string = ""
        for app in self.approaches:
            string = string + self.ds.name+","+app.name+","+app.statistic_rates.toString()+"\n"
        return string
    
    def isNormalDistributionAll(self):
        cont = 0
        for app in self.approaches:
            if app.IsNormalDistribution:
                cont+=1
        return cont == len(self.approaches)
    
    def __str__(self):
        csvtable = self.ds.name+","
        for app in self.approaches:
            csvtable = csvtable + app.ErrorRate + "," + app.StandardDeviation + "," + app.rank + ","
            if not app.IsApproachBase:
                csvtable = csvtable + app.pvalue + "," + app.h0 + ","
            csvtable = csvtable + "\n"
        return csvtable
    
    def copy(self):
        my_copy = ApproachCollection(self.ds)
        my_copy.approaches = self.approaches.copy()
        return my_copy    
    