from argparse import Namespace
import os, math, sys
import scipy.io
from Datasets.dataset import Dataset
from eMODiTS.Scheme import Scheme
import pandas as pd
import numpy as np
from Utils.utils import to_paired_value
from RegressionMeasures.regression_measures import RegressionMeasures

args = {"ff": 0,
        "train_rep": 'all',
       "pm":0.2,
       "evaluation_measure":"MD"}
opts = Namespace(**args)

def load_cuts_from_matlab(ds, cuts_list):
    cuts = {}
    wordcuts = set(cuts_list[:,0])
    word_cuts = to_paired_value(sorted(list(wordcuts.union({1,ds.dimensions[1]-1}))))
    alphs = cuts_list[0,1:len(cuts_list)]
    for i in range(0,len(word_cuts)):
        alphs_cuts = cuts_list[i,1:len(cuts_list[i])]
        alphs = set(alphs_cuts[~np.isnan(alphs_cuts)])
        alphscuts = to_paired_value(sorted(list(alphs.union({ds.limites[0], ds.limites[1]+1}))))
        cuts[str(word_cuts[i])] = list(alphscuts)
    return cuts

def get_fitness_functions(ds, location):
    filename = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"+location+"/MODiTS/"+ds.name+"/"+ds.name+"_MODiTS.mat"
    surr_results = scipy.io.loadmat(filename)
    surrogate_front = np.array(surr_results['AccumulatedFrontFitness'])
    orig_front = []
    for i in range(0,surrogate_front.shape[0]):
        cuts = load_cuts_from_matlab(ds, surr_results["FrontIndividual"+str(i)])
        sch = Scheme(ds=ds, cuts=cuts, options=opts)
        sch.evaluate()
        orig_front.append(sch.fitness_functions.values)
    return np.array(orig_front), surrogate_front

def export_prediction_power(locations, ids):
    
    orig_all = np.empty((0, 3), float)
    surr_all = np.empty((0, 3), float)
    for loc in locations:
        data = []
        for i in ids:
            ds = Dataset(i, '_TRAIN', False)
            row = []
            row.append(ds.name)
            orig, surr = get_fitness_functions(ds, location=loc)
            
            #print(orig, surr)
            
            orig_all = np.concatenate((orig_all, orig), axis=0)
            surr_all = np.concatenate((surr_all, surr), axis=0)
            
            for i in range(0,orig.shape[1]):
                ff_orig = pd.Series(orig[:,i])
                ff_surr = pd.Series(surr[:,i])
                measures = RegressionMeasures(observed=ff_orig, predicted=ff_surr)
                measures.compute()
                row.append(measures.values["R"])
                row.append(measures.values["R2"])
                row.append(measures.values["MSE"])
                row.append(measures.values["RMSE"])
                row.append(measures.values["MD"])
            data.append(row)

    
        headers2 = ['Entropy Orig', 'Complexity Orig', 'InfoLoss Orig', 
                'Entropy Surr', 'Complexity Surr', 'InfoLoss Surr']#, 'MAPE InfoLoss']
        fitness = np.concatenate((orig_all,surr_all), axis=1)
        overall_entropy = RegressionMeasures(observed=fitness[:,0],predicted=fitness[:,3])
        overall_entropy.compute()
        print('MSE Entropy:',overall_entropy.values["MSE"])
        print('RMSE Entropy:',overall_entropy.values["RMSE"])
        print('R2 Entropy:',overall_entropy.values["R2"])
        print('R Entropy:',overall_entropy.values["R"])
        print('MD Entropy:',overall_entropy.values["MD"])
        overall_complexity = RegressionMeasures(observed=fitness[:,1],predicted=fitness[:,4])
        overall_complexity.compute()
        print('MSE Complexity:',overall_complexity.values["MSE"])
        print('RMSE Complexity:',overall_complexity.values["RMSE"])
        print('R2 Complexity:',overall_complexity.values["R2"])
        print('R Complexity:',overall_complexity.values["R"])
        print('MD Complexity:',overall_complexity.values["MD"])
        overall_infoloss = RegressionMeasures(observed=fitness[:,2],predicted=fitness[:,5])
        overall_infoloss.compute()
        print('MSE Information Loss:',overall_infoloss.values["MSE"])
        print('RMSE Information Loss:',overall_infoloss.values["RMSE"])
        print('R2 Information Loss:',overall_infoloss.values["R2"])
        print('R Information Loss:',overall_infoloss.values["R"])
        print('MD Complexity:',overall_infoloss.values["MD"])
        
        DF2 = pd.DataFrame(fitness)
        if "NNDTW" in loc:
            filename = "KNNDTW"
        elif "RBF" in loc:
            filename = "RBFN"
        elif "SVR" in loc:
            filename = "SVR"
    
        directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"+loc+"/Correlations/"
        if not os.path.isdir(directory):
            os.mkdir(directory)
        with open(directory+"/ModelsResults_"+filename+".csv", "w") as file_csv:
            DF2.to_csv(file_csv, header=headers2)
        
        #print(data)
        headers = ['Dataset', 'R Entropy', 'R2 Entropy', 'MSE Entropy', 'RMSE Entropy', 'MD Entropy', 
                'R Complexity', 'R2 Complexity', 'MSE Complexity','RMSE Complexity', 'MD Complexity',
                'R InfoLoss', 'R2 InfoLoss', 'MSE InfoLoss', 'RMSE InfoLoss', 'MD InfoLoss']
        DF = pd.DataFrame(data)
        directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"+loc+"/Correlations/"
        if not os.path.isdir(directory):
            os.mkdir(directory)
        with open(directory+"/CorrelationResults_"+filename+".csv", "w") as file2_csv:
            DF.to_csv(file2_csv, header=headers)
    
#export_prediction_power("e15p100g300_u5")
#if __name__ == "main":
#    locations = ["e15p100g300_sMODiTS/1NNDTW","e15p100g300_sMODiTS/7NNDTW","e15p100g300_sMODiTS/9NNDTW"]
#    ids = [2,7,10,12,13,14,15,16,17,18,20,21,22,24,26,31,37,38,41,44,45,46,47,48,53,55,56,57,58,64,65,68,70,71,72,73,74,75,80,81]
#    export_prediction_power(locations=locations,ids=ids)


""" import os, math
import scipy.io
from Datasets.dataset import Dataset
from eMODiTS.Scheme import Scheme
#from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np
#from sklearn.metrics import mean_squared_error, r2_score
from Utils.utils import to_paired_value
from RegressionMeasures.regression_measures import RegressionMeasures
from argparse import Namespace

# Pearson correlation
# -1 indicates a perfectly negative linear correlation
# 0 indicates no linear correlation
# 1 indicates a perfectly positive linear correlation

#y_true = [0.98, 0.7, 0.66]
#y_true = [1, 2, 3]
#y_pred = [0.95, 0.2, 0.5]
#y_pred = [3, 2, 1]

#print(mean_squared_error(y_true, y_pred))

def get_fitness_functions(ds, location):
    filename = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"+location+"/MODiTS/"+ds.name+"/"+ds.name+"_MODiTS.mat"
    surr_results = scipy.io.loadmat(filename)
    surrogate_front = np.array(surr_results['AccumulatedFrontFitness'])
    orig_front = []
    args = {"knn_k":3, 
        "task": 'regression', 
        "dist_metric": 'dtw', 
        "dtw_method": 'sakoechiba',
       "dtw_sakoechiba_w": 10,
       "exec_type": 'cpu',
        "dtw_dist": 'square',
        "ff": 0,
        "train_rep": 0,
        "dist_t": 10,
       "dtw_precost": False,
       "dtw_return_cost": False,
       "dtw_return_accum": False,
       "dtw_return_path":False}
    #print(args)

    my_options = Namespace(**args)
    for i in range(0,surrogate_front.shape[0]):
        #cuts = load_cuts_from_matlab(ds, surr_results["FrontIndividual"+str(i)])
        #sch = Scheme(ds=ds, cuts=cuts)
        #sch.evaluate()
        sch = Scheme(ds=ds, options=my_options)
        sch.reset()
        sch.load_from_lists(cuts=surr_results["FrontIndividual"+str(i)], ff=[])
        sch.evaluate()
        orig_front.append(sch.fitness_functions.values)
    return np.array(orig_front), surrogate_front
 
def export_prediction_power(iDS,location):
    data = []
    orig_all = np.empty((0, 3), float)
    surr_all = np.empty((0, 3), float)
    print("*",location)
    
    for d in range(0,len(iDS)):
        ds = Dataset(iDS[d], '_TRAIN', False)
        row = []
        print("---",ds.name)
        row.append(ds.name)
        orig, surr = get_fitness_functions(ds, location=location)
        #print(orig, surr)
        orig_all = np.concatenate((orig_all, orig), axis=0)
        surr_all = np.concatenate((surr_all, surr), axis=0)
        
        for i in range(0,orig.shape[1]):
            #ff_orig = pd.Series(orig[:,i])
            #ff_surr = pd.Series(surr[:,i])
            measures = RegressionMeasures(observed=orig[:,i], predicted=surr[:,i])
            #r = ff_orig.corr(ff_surr)
            #mse = mean_squared_error(ff_orig, ff_surr)
            #r2 = r2_score(ff_orig, ff_surr)
            #rmse = math.sqrt(mse)
            #r = measures.r()
            r2 = measures.R2()
            mse = measures.MSE()
            md = measures.md()
            #row.append(r)
            row.append(r2)
            row.append(mse)
            row.append(md)
            #row.append(mape)
        data.append(row)
    
    
    headers2 = ['Entropy Orig', 'Complexity Orig', 'InfoLoss Orig', 
               'Entropy Surr', 'Complexity Surr', 'InfoLoss Surr']#, 'MAPE InfoLoss']
    fitness = np.concatenate((orig_all,surr_all), axis=1)
    measuresModelE = RegressionMeasures(observed=fitness[:,0], predicted=fitness[:,3])
    print('R2 Entropy:',measuresModelE.R2())
    print('MSE Entropy:',measuresModelE.MSE())
    print('dj Entropy:',measuresModelE.md())
    measuresModelC = RegressionMeasures(observed=fitness[:,1], predicted=fitness[:,4])
    print('R2 Complexity:',measuresModelC.R2())
    print('MSE Complexity:',measuresModelC.MSE())
    print('dj Complexity:',measuresModelC.md())
    measuresModelIF = RegressionMeasures(observed=fitness[:,2], predicted=fitness[:,5])
    print('R2 Information Loss:',measuresModelIF.R2())
    print('MSE Information Loss:',measuresModelIF.MSE())
    print('dj Information Loss:',measuresModelIF.md())
    
    DF2 = pd.DataFrame(fitness)
    directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"+location+"/Correlations/"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    DF2.to_csv(directory+"/ModelsResults_"+location+".csv", header=headers2)
    
    #print(data)
    headers = ['Dataset', 'R2 Entropy', 'MSE Entropy', 'dj Entropy', 
               'R2 Complexity', 'MSE Complexity','dj Complexity', 
               'R2 InfoLoss', 'MSE InfoLoss', 'dj InfoLoss']
    DF = pd.DataFrame(data)
    directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"+location+"/Correlations/"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    DF.to_csv(directory+"/CorrelationResults_"+location+".csv", header=headers)

#if __name__ == "__main__":
#    iDS = [38,70,20,7,64,65,48,74,31,16,55,44,45,56,17,21,22,75,26,47,24,58,18,2,10]
#    export_prediction_power(iDS, "e15p100g300_gu30_gi1_m2_k1_w0_dtw_fast_None_updStr1") """