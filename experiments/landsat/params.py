import collections

seed = 0
odir = "results"
#methods = ["hugewood", "subsetwood", "sk", "h2"]
methods = ["hugewood"]

datasets = collections.OrderedDict()
datasets['landsat'] = {'train_sizes':[i*1000000 for i in [10,20,30,40]]}

parameters = collections.OrderedDict()
#parameters['ert'] = {'n_estimators':4,
#                     'max_features':None, 
#                     'bootstrap':False, 
#                     'tree_type':'randomized', 
#                     'n_jobs':4}
parameters['rf'] = {'n_estimators':12,
                    'max_features':"sqrt", 
                    'bootstrap':True, 
                    'tree_type':'standard', 
                    'n_jobs':4}

parameters_hugewood = collections.OrderedDict()

for key in parameters:
    
    param_hugewood = {}
    param_hugewood['param_wood'] = parameters[key]
    param_hugewood['n_estimators'] = 3
    param_hugewood['n_estimators_bottom'] = 4
    
    parameters_hugewood[key] = param_hugewood
