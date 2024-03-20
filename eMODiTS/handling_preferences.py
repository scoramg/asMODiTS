import scipy.io

class ReferencePoint:
    def __init__(self, ds, front = None, point_type = 0, usingTest = False):
        self.front = front
        self.type = point_type
        self.usingTest = usingTest
        self.ds = ds
        
    def __del__(self):
        del(self.front)
        del(self.type)
        del(self.usingTest)
        del(self.ds)
        
    def getName(self):
        if self.usingTest:
            return "CTT"
        else:
            return "CTV"
        
    def getSelection(self):
        ind = -1
        minErr = float('inf')
    
    '''public int getSelection() {
        int ind = -1;
            
        double minErr = Double.POSITIVE_INFINITY;

        IScheme indAux;
        if (this.individual_type == 0){
            indAux = new Individuals.Proposal.MOScheme();
        }
        else {
            if (this.individual_type == 1){
                indAux = new Individuals.Proposal.MOScheme();
            } else {
                indAux = new Individuals.PEVOMO.MOScheme();
            }
        }

        Map<String, MLArray> mlArrayRetrived = mfr.getContent();

        for(int j=0;j<this.front.length;j++){
            indAux.empty();
            MLArray f = mlArrayRetrived.get("FrontIndividual"+j);
            double[][] individual = ((MLDouble) f).getArray();
            indAux.add(individual);
//                indAux.setDiscretization(ds);
            indAux.Classify(ds, usingTEST, "train");
//            StringBuilder sb = new StringBuilder();
//            sb.append("j:").append(j).append(", indAux.getErrorRate():").append(indAux.getErrorRate()).append(", minErr:").append(minErr);
            if (indAux.getErrorRate() < minErr){
                ind = j;
                minErr = indAux.getErrorRate();
//                sb.append(", ind:").append(ind);
            }
//            System.out.println(sb.toString());
        }
        return ind;
    }'''