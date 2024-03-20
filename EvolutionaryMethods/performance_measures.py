import numpy as np
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.gd import GD

def dominates(x, y):
    ''' dominate1 = 0 
    dominate2 = 0

    flag = 0

    for i in range(0,len(x)):
        if x[i] < y[i]:
            flag = -1
        elif x[i] > y[1]:
            flag = 1
        else:
            flag = 0
            
        if flag == -1:
            dominate1 = 1
        
        if flag == 1:
            dominate2 = 1

    if dominate1 == dominate2:
        dominance = False

    if dominate1 == 1:
        dominance = True #f1 dominates f2
        
    if dominate2 == 1:
        dominance = False '''
    dominance = np.all(x<=y) and np.any(x<y)
    return dominance

def cover_measure(frontA, frontB):
    cont_a = 0
    cont_b = 0
    dominated = set()
    for i in range(0,len(frontA)):
        for j in range(0,len(frontB)):
            if dominates(frontA[i], frontB[j]):
                dominated.add(j)
    return len(dominated)/len(frontB)

def get_hv_ratio(reference_point, reference_front, compared_front):
    ind = Hypervolume(ref_point=reference_point)
    return ind.do(compared_front)/ind.do(reference_front)

def get_gd(reference_front, compared_front):
    ind = GD(pf=reference_front)
    return ind.do(F=compared_front)


