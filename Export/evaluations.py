import os
import scipy.io
from Datasets.dataset import Dataset

def compute_evaluation_ratio(dataset_name):
    filename = os.path.dirname(os.path.realpath(__file__))+"/Results/e15p100g300_u5/MODiTS/"+dataset_name+"/"+dataset_name+"_MODiTS.mat"
    sch = scipy.io.loadmat(filename);
    return sch['evaluations'];
 
#iDS = [1,2,7,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,31,37,38,41,44,45,46,47,48,53,55,56,57,58,64,65,67,68,70,71,72,73,74,75,80,82,85]
#iDS_ignored = [51,60,49,50,83,84,9,36,33,86,87,88,89,90,91,92,93,94,95];
iDS = [38,70,65,64,16,17,18,45,47,56,58,74,48,20,44,7,75,24,26,22,21,55,31,2,10]
#print(len(iDS))
eMODiTS_evals = 451500.0
sum_evals = 0
no_ds = 0
#for i in range(1,95):
for i in range(0,len(iDS)):
    #if i not in iDS_ignored:
    ds = Dataset(i, '_TRAIN', False)
    ratio = compute_evaluation_ratio(ds.name)  
    sum_evals += ratio
    no_ds+=1
    print(ds.name,",",ratio,",",(abs(ratio-eMODiTS_evals) / eMODiTS_evals)*100)
print("No. Datasets=",no_ds,", promedio=",(sum_evals/no_ds))