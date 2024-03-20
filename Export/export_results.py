import os, sys
sys.path.append(os.path.abspath(os.path.join('.', 'src')))
from eMODiTS.Scheme import Scheme
from Datasets.dataset import Dataset
from eMODiTS.handling_preferences import ReferencePoint

class Methods:
    def __init__(self, ClassificationType=0, Location="", type_selection=""):
        self.ClassificationType = ClassificationType
        self.Location = Location
        self.type_selection = type_selection
    
    def getBestEMODiTS(self, ds):
        selector = ReferencePoint(ds=ds)
        filename = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/Results/"+self.Location+"/MODiTS/"+\
            ds.name + "/" +  ds.name + "_MODiTS_" + selector.getName() + "_B.mat"
        sch = Scheme(ds, cuts = {}, filename=filename, idfunctionsconf=0)
        
        if self.ClassificationType == 1: # CV on original
            sch.classify(UsingTest=False, set_type="original", UsingDiscrete=True)
        elif self.ClassificationType == 2: #CV on test
            sch.classify(UsingTest=False, set_type="test", UsingDiscrete=True)
        elif self.ClassificationType == 3: #Train-Test without test subdivision   
            sch.classify(UsingTest=True, set_type="WithoutCV", UsingDiscrete=True)
        elif self.ClassificationType == 4: #Train-Test with test subdivision   
            sch.classify(UsingTest=True, set_type="WithCV", UsingDiscrete=True)
        elif self.ClassificationType == 5: #CV on train  
            sch.classify(UsingTest=False, set_type="train", UsingDiscrete=True)
        else:
            sch.classify(UsingTest=False, set_type="original", UsingDiscrete=True)
        
        sch.ExportGraph("MODiTS", self.Location, selector.getName())
        sch.ExportClassificationResultsByFolds(self.Location, self.type_selection)
        return sch

class Export:
    def __init__(self):
        pass