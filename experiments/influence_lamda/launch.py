import os
import params

seeds = [0,1,2,3]
odir = params.odir
methods = params.methods

for method in methods:
    for dkey in params.datasets.keys():
        for train_size in params.datasets[dkey]['train_sizes']:
            for lamcrit in params.lamcrits:
                for seed in seeds:
                    for key in params.parameters:
                        print("Processing method %s with data set %s, train_size %s, lamcrit %s, seed %s, and key %s ..." % (str(method), str(dkey), str(train_size), str(lamcrit), str(seed), str(key)))
                        cmd = "python " + method + ".py --dkey %s --train_size %i --lamcrit %f --seed %i --key %s" % (dkey, train_size, lamcrit, seed, key)
                        print(cmd)
                        os.system(cmd)
