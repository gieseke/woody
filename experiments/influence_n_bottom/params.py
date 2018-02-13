import collections

odir = "results"
methods = ["hugewood_1K", "hugewood_10K", "hugewood_75K"]

n_estimators_bottoms = [1,4,12,24]

datasets = collections.OrderedDict()
datasets['covtype'] = {'train_sizes':[100000, 150000, 200000, 250000, 300000, 350000, 400000]}

parameters = collections.OrderedDict()
#parameters['ert'] = {'n_estimators':4,
#                     'max_features':None, 
#                     'bootstrap':False, 
#                     'tree_type':'randomized', 
#                     'n_jobs':4}
parameters['rf'] = {'n_estimators':24,
                    'max_features':"sqrt", 
                    'bootstrap':True, 
                    'tree_type':'standard', 
                    'n_jobs':4}

parameters_hugewood = collections.OrderedDict()

for key in parameters:
    
    param_hugewood = {}
    param_hugewood['param_wood'] = parameters[key]
    # set in hugewood*.py
    #param_hugewood['n_estimators'] = 6
    #param_hugewood['n_estimators_bottom'] = 4
    
    parameters_hugewood[key] = param_hugewood
