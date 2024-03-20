from Functions.confusion_matrix import init_dictionary
from sklearn.metrics import mean_squared_error

class FitnessFunction:
        
    def __init__(self, idconfiguration=0):
        self.values = []
        self.idconfiguration = idconfiguration
    
    @property
    def size(self):
        return len(self.values)
    
    @property
    def no_functions(self):
        if self.idconfiguration == 0:
            return 3
        return 0
    
    @property
    def functions_name(self):
        if self.idconfiguration == 0:
            return ["Entropy", "Complexity", "InfoLoss"]
        return 0
    
    def evaluate(self, scheme):
        if self.idconfiguration == 0:
            self.values.append(self.entropy(scheme))
            self.values.append(self.complexity(scheme))
            self.values.append(self.InfoLoss(scheme))
        
    @property
    def _no_functions(self):
        return len(self.values)
            
    def entropy(self, scheme):
        import math
        matrix_entropy = {}
        nonempty_entropy = {}
        sum_nonempty_count = 0
        sum_entropy = 0
        accuracy = 0
        for string, counts in scheme.confusion_matrix.matrix.items():
            probs = init_dictionary(scheme.confusion_matrix.clases) 
            sum_prob = 0
            for klass, value in counts.items():
                prob = int(value)/scheme.confusion_matrix.row_total[string]
                if prob == 0:
                    prob = 1
                probs[klass] = prob * math.log(1/prob)
                sum_prob += probs[klass]
                sum_entropy += probs[klass]
            matrix_entropy[string]=probs
            if sum_prob > 0:
                nonempty_entropy[string] = probs
                for _,c in scheme.confusion_matrix.matrix[string].items():
                    sum_nonempty_count += int(c)

        if len(nonempty_entropy)>0:
            pr = (sum_entropy * sum_nonempty_count)/len(nonempty_entropy)
            accuracy = 1 - (1/(pr+1));
        return accuracy

    def complexity(self, scheme):
        complexity = 0
        if (len(scheme.confusion_matrix.matrix) - len(scheme.confusion_matrix.clases)) < 0:
            complexity = scheme.ds.dimensions[0] - (len(scheme.confusion_matrix.matrix) - len(scheme.confusion_matrix.clases))
        else:
            complexity = len(scheme.confusion_matrix.matrix) - len(scheme.confusion_matrix.clases)
        return complexity / (scheme.ds.dimensions[0] + len(scheme.confusion_matrix.clases) - 1)

    def InfoLoss(self, scheme):
        return mean_squared_error(scheme.ds.data_norm[:,1:],scheme.ds_discrete_reconstructed[:,1:])
    
    def copy(self):
        ff = FitnessFunction(self.idconfiguration)
        ff.values = self.values.copy()
        return ff
        
