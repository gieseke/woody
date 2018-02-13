import collections

seed = 0
odir = "results"
methods = ["hugewood"]

datasets = collections.OrderedDict()
datasets['landsat'] = {'train_sizes':[250000000, 500000000, 750000000, 1000000000]}

parameters = collections.OrderedDict()
#parameters['ert'] = {'n_estimators':4,
#                     'max_features':None, 
#                     'bootstrap':False, 
#                     'tree_type':'randomized', 
#                     'n_jobs':4}
parameters['rf'] = {'n_estimators':4,
                    'max_features':"sqrt", 
                    'bootstrap':True, 
                    'tree_type':'standard', 
                    'n_jobs':4}

parameters_hugewood = collections.OrderedDict()

for key in parameters:
    
    param_hugewood = {}
    param_hugewood['param_wood'] = parameters[key]
    param_hugewood['n_estimators'] = 1
    param_hugewood['n_estimators_bottom'] = 4
    
    parameters_hugewood[key] = param_hugewood
