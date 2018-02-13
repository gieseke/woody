import os
import params

seeds = [0,1,2,3]
odir = params.odir
methods = params.methods

for method in methods:
    for dkey in params.datasets.keys():
        for train_size in params.datasets[dkey]['train_sizes']:
            for n_bottom in params.n_estimators_bottoms:
                for seed in seeds:
                    for key in params.parameters:
                        print("Processing method %s with data set %s, train_size %s, n_bottom %s, seed %s, and key %s ..." % (str(method), str(dkey), str(train_size), str(n_bottom), str(seed), str(key)))
                        cmd = "python " + method + ".py --dkey %s --train_size %i --n_bottom %f --seed %i --key %s" % (dkey, train_size, n_bottom, seed, key)
                        print(cmd)
                        os.system(cmd)
