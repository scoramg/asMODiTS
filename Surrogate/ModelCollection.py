from Surrogate.RBF import RBF
from Surrogate.KNN import KNN
from Surrogate.SVR import SVR
from Surrogate.RBFNN import RBFNN
from eMODiTS.Population import Population

class ModelCollection:
    def __init__(self, options=None):
        self.options = options
        self.models = []
        self.name = None
        
    @property
    def is_same_model(self):
        return all(i == self.options.model[0] for i in self.options.model)
        
    @property
    def no_models(self):
        return len(self.models)
        
    @property
    def training_number(self):
        no_training = 0
        for t in range(0,len(self.models)):
            no_training += self.models[t].training_number
        return no_training
        
    def create(self):
        for i in range(0, len(self.options.model)):
            model = eval(self.options.model[i].name)(id_model=i, options=self.options)
            self.models.append(model.copy())
            if self.name:
                self.name = self.name + "_" + model.get_name().upper()
            else:
                self.name = model.get_name().upper()
                
    def update(self, **kwargs):
        included = 0
        no_eval = 0
        g = kwargs['g']
        for i in range(0,self.no_models):
            if ((g+1) % self.models[i].gen_upd) == 0:
                included, no_eval = self.models[i].update(kwargs)
                inc = round(self.models[i].factor_act*included,0)
                if inc > 0:
                    self.models[i].gen_upd += inc
                else:
                    self.models[i].gen_upd += 1
        return included, no_eval
            
    def train(self, initial_training_set):
        for i in range(0,self.no_models):
            self.models[i].fit(initial_training_set.copy())
            self.models[i].train()
            
    def restore(self, ds, checkpoint):
        for i in range(0,self.no_models):
            self.models[i].restore(ds, checkpoint["Model"+str(i)])
            
    def create_checkpoint(self):
        checkpoint = {}
        for i in range(0,self.no_models):
            checkpoint["Model"+str(i)] = {}
            checkpoint["Model"+str(i)]["training_set"] = self.models[i].training_set.get_individuals_cuts()
            if self.options.model[i].ue == "archive":
                checkpoint["Model"+str(i)]["archive"] = self.models[i].archive.create_checkpoint()
            checkpoint["Model"+str(i)]["gen_upd"] = self.models[i].gen_upd
            checkpoint["Model"+str(i)]["factor_act"] = self.models[i].factor_act
            checkpoint["Model"+str(i)]["is_trained"] = self.models[i].is_trained
            checkpoint["Model"+str(i)]["is_fitted"] = self.models[i].is_fitted
            checkpoint["Model"+str(i)]["training_number"] = self.models[i].training_number
        return checkpoint
            
    def load_matlab(self, data, ds):
        for i in range(0,len(data)):
            model_data = data['Model'+str(i)]
            if model_data["name"] == self.models[i].class_name:
                model = eval(model_data["name"])(id_model=model_data["id_model"], options=self.options)
                training_set = Population(_ds=ds, options=self.options)
                training_set.load_matlab(model_data["training_set"])
                model.fit(training_set.copy())
            
    def export_matlab(self):
        data = {}        
        for i in range(0,self.no_models):
            data['Model'+str(i)] = self.models[i].export_matlab()
        return data