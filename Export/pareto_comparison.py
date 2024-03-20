import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#import numpy as np
import pandas as pd
#from pymoo.problems import multi
from pymoo.config import Config
from EvolutionaryMethods.performance_measures import cover_measure, get_hv_ratio, get_gd
from scipy.io import savemat, loadmat
from Datasets.dataset import Dataset

#from pymoo.algorithms.moo.nsga2 import NonDominatedSorting

Config.show_compile_hint = False


def get_fronts(dataset_name, base, location):
    filename = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"+location+"/MODiTS/"+dataset_name+"/"+dataset_name+"_MODiTS.mat"
    sch = loadmat(filename)
    accumulate_sMODiTS = sch['AccumulatedFrontFitness']
    
    filename = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"+base+"/MODiTS/"+dataset_name+"/"+dataset_name+"_MODiTS.mat"
    sch = loadmat(filename)
    accumulate_eMODiTS = sch['AccumulatedFrontFitness']
    return accumulate_eMODiTS, accumulate_sMODiTS

def get_pareto_comparison(iDS, base, location):
    
    model = location.split("_")[5]
    
    ratios = []
    gds = []
    cover_emodits = []
    cover_smodits = []
    datasets = []
    ci = []
    
    col1 = "Dataset"
    col2 = "HV Ratio" #coello2007evolutionary
    col3 = "GD" #knowlessGD
    col4 = "C(eMODiTS,"+model+")" #coverage
    col5 = "C("+model+",eMODiTS)" #coverage
    col6 = "CI" #coverage
    
    for i in range(0,len(iDS)):
        #if i not in iDS_ignored:
        ds = Dataset(iDS[i], '_TRAIN', False)
        accumulate_eMODiTS, accumulate_sMODiTS = get_fronts(ds.name, base, location)
        max_e = max(accumulate_eMODiTS[:,2])
        max_s = max(accumulate_sMODiTS[:,2])
        ref_point = [1,1,max_e+max_s] 
        datasets.append(ds.name)
        ratios.append(get_hv_ratio(ref_point, accumulate_eMODiTS, accumulate_sMODiTS))
        gds.append(get_gd(accumulate_eMODiTS, accumulate_sMODiTS))
        c_emodits = cover_measure(accumulate_eMODiTS, accumulate_sMODiTS)
        c_smodits = cover_measure(accumulate_sMODiTS, accumulate_eMODiTS)
        cover_emodits.append(c_emodits)
        cover_smodits.append(c_smodits)
        ci.append(c_emodits-c_smodits)
        #data.append([i, ratio, gd, SC_eMODiTS, SC_sMODiTS])
    directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"+location+"/ParetoAnalysis/"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    data = pd.DataFrame({col1:datasets,col2:ratios,col3:gds,col4:cover_emodits,col5:cover_smodits,col6:ci})
    data.to_excel(directory+'performance_pareto.xlsx', sheet_name='sheet1', index=False)
    #table['table']=data
    #savemat("performance_pareto.mat", table)
    
def get_pareto_per_method(iDS, base, location):
    mat = {}
    model = location.split("_")[5]
    ds = Dataset(iDS, '_TRAIN', False) #ItalicPowerDemand
    accumulate_eMODiTS, accumulate_sMODiTS = get_fronts(ds.name, base, location)
    #nds = NonDominatedSorting()
    #nds.do(F)
    mat[model] = accumulate_sMODiTS
    mat['EC_'+model] = accumulate_sMODiTS[:,:2]
    mat['EL_'+model] = accumulate_sMODiTS[:,[0,2]]
    mat['CL_'+model] = accumulate_sMODiTS[:,1:3]
    mat['eMODiTS'] = accumulate_eMODiTS
    mat['EC_eMODiTS'] = accumulate_eMODiTS[:,:2]
    mat['EL_eMODiTS'] = accumulate_eMODiTS[:,[0,2]]
    mat['CL_eMODiTS'] = accumulate_eMODiTS[:,1:3]
    
    directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"+location+"/ParetoAnalysis/"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    
    savemat(directory+"pareto_fronts_"+ds.name+".mat", mat, long_field_names=True)


def execute():
    iDS = [38, 70, 20, 7, 64, 65, 48, 74, 31, 16, 55, 44, 45, 56, 17, 21, 22, 75, 26, 47, 24, 58, 18, 2, 10]
    base = "e15p100g300"
    locations = ["CIAPP/e15p100g300_gu30_gi1_m0_k1_w0_dtw_fast_None_updStr1",
                "CIAPP/e15p100g300_gu30_gi1_m0_k3_w0_dtw_fast_None_updStr1",
                "CIAPP/e15p100g300_gu30_gi1_m0_k5_w0_dtw_fast_None_updStr1",
                "CIAPP/e15p100g300_gu30_gi1_m0_k7_w0_dtw_fast_None_updStr1",
                "CIAPP/e15p100g300_gu30_gi1_m0_k9_w0_dtw_fast_None_updStr1"]

    for location in locations:
        print("*", location)
        get_pareto_comparison(iDS, base, location)
        for i in range(len(iDS)):
            get_pareto_per_method(iDS[i], base, location)
            
if __name__ == "__main__":
    execute()
    #print(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))