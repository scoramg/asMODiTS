import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Datasets.dataset import Dataset
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt


__SAVEPATH__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"

def get_time(ds, location):
    filename = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"+location+"/MODiTS/"+ds.name+"/"+ds.name+"_MODiTS.mat"
    surr_results = scipy.io.loadmat(filename)
    time = surr_results['time']
    return time[0]

def get_evaluations(ds, folder, location):
    no_eval = 0
    for e in range(0,15):
        filename = __SAVEPATH__+folder+location+"/MODiTS/"+ds.name+"/"+ds.name+"_exec"+str(e)+"_MODiTS.mat"
        surr_results = scipy.io.loadmat(filename)
        no_eval += surr_results['evaluations'][0][0]
    return no_eval

def plot(ids, folder, locations, colors, legends, function):
    evals = {}
    #bd = []
    orig_eval = [100*300*15]*len(ids)
    fig = plt.figure(figsize=(14, 8))
    plt.title("Number of evaluations in the original models")
    plt.xlabel('Datasets')
    plt.ylabel('Number of evaluations')
    plt.plot(orig_eval, "k-", label="eMODiTS")
    for l in range(0,len(locations)):
        evals[locations[l]] = []
        for i in ids:
            ds = Dataset(i, '_TRAIN', False)
            #bd.append(ds.name)
            evals[locations[l]].append(eval(function)(ds, folder, locations[l]))
        #print(evals[locations[l]])
        plt.plot(evals[locations[l]], colors[l]+"-", label=legends[l])
    plt.xticks(range(0,len(ids)))
    plt.legend()
    #plt.show()
    print(__SAVEPATH__+folder+'ComputationalCost.tiff')
    print(__SAVEPATH__+folder+"ComputationalCost.tex")
    import tikzplotlib
    tikzplotlib.save(figure=fig,filepath=__SAVEPATH__+folder+"ComputationalCost.tex")
    fig.savefig(__SAVEPATH__+folder+'ComputationalCost.tiff', dpi=300)

""" def graficar():
    iDS = [38,70,65,64,47,16]
    time_m2 = []
    time_m3 = []
    for i in range(0,len(iDS)):
        ds = Dataset(iDS[i], '_TRAIN', False)
        time_m2.append(get_time(ds, 'e1p100g300_u5_m2'))
        time_m3.append(get_time(ds, 'e1p100g300_u5_m3'))
    print(time_m2)
    print(time_m3)
    plt.figure(figsize=(14, 8))
    plt.plot(time_m2, "r-", label='DTW sin LBKeogh')
    plt.plot(time_m3, "b-", label='DTW con LBKeogh')
    plt.legend()
    plt.xticks(list(range(0,17)), ['CBF','DistalPhalanxOutlineAgeGroup','ECG200','ECGFiveDays','FaceAll','FacesUCR','ItalyPowerDemand','MedicalImages','MiddlePhalanxOutlineAgeGroup','MiddlePhalanxTW','MoteStrain','ProximalPhalanxTW','SonyAIBORobotSurface1','SonyAIBORobotSurface2','SwedishLeaf','SyntheticControl','TwoPatterns'],rotation=20) """

if __name__ == "__main__":
    colors = {}
    ids = [7,16,20,22,24,26,38,44,45,47,48,58,64,65,68,70,75]
    folder = "e15p100g300_sMODiTS/"
    locations = ["BU10_1NNDTW(AR_5_43)_5NNDTW(AR_5_58)_3NNDTW(AR_5_88)",
                "BU10_RBFNNDTW(AR_5_82_8_50_9)_RBFNNDTW(AR_5_45_8_100_9)_RBFNNDTW(AR_5_46_7_150_16)",
                "BU10_SVRGAK(AR_5_10_6_0.0)_SVRGAK(AR_5_10_10_0.0)_SVRGAK(AR_5_10_9_0.0)"]
    colors = ["r","b","g","c","m"]
    legends = ["$asMODiTS_{KNN}$", "$asMODiTS_{RBFN}$", "$asMODiTS_{SVR}$"]
    #plot(ids, locations, colors, legends, "get_evaluations")
    plot(ids, folder, locations, colors, legends, "get_evaluations")
    
    
    
    #Oficio dirigido a la DGI, notificando que renuncié a la beca y que se ha informado a conacyt con vo. bo. de dr efren
    
    
