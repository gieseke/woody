from sklearn.metrics import accuracy_score

metrics = {"accuracy": accuracy_score}

def evaluate(preds, y, results, prefix, verbose=1):
    
    for key in metrics.keys():
        res = metrics[key](y, preds)
        results[prefix + "_" + key] = res
        if verbose > 0:
            print(prefix + " " + key + ":\t" + str(res))
            