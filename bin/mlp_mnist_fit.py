#! /usr/bin/env python3

import os
import json

import argh
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

from explore_ml.utils.mnist_utils import *
from explore_ml.utils.data_utils import *
from explore_ml.utils.param_utils import generate_params

def train_model(params, train_data, train_labels):
    nn = MLPClassifier(**params)
    nn.fit(flatten_data(train_data), train_labels)
    return nn

def validate_model(model, valid_data, valid_labels):
    return model.score(flatten_data(valid_data), valid_labels)

def run_models(parameter_space,
               train_data,
               train_labels,
               valid_data,
               valid_labels,
               output_dir):
    best_i = None
    best_acc = 0
    best_params = None
    for i,params in enumerate(generate_params(parameter_space)):
        print(params)
        model = train_model(params, train_data, train_labels)
        valid_acc = validate_model(model, valid_data, valid_labels)

        stats = {}
        stats["params"] = model.get_params()
        stats["validation_accuracy"] = valid_acc

        json_out = os.path.join(output_dir, "model_%d_stats.json" % (i))
        json.dump(stats, open(json_out,"w"))

        model_out = os.path.join(output_dir, "model_%d.pickle" % (i))
        joblib.dump(model, open(model_out,"wb"))

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_i = i
            best_params = params

    print("Model %d had highest validation accuracy: %f" % (best_i,best_acc))
    print("Parameters: %s" % (json.dumps(best_params,sort_keys=True,indent=4)))




    
@argh.arg("-d","--data-pickle",required=True)
@argh.arg("-p","--parameter-space",required=True,help = "JSON file "
          "specifying parameter space to explore, see param_utils.py "
          "for example.")
def main(data_pickle = None,
         parameter_space = None,
         output_dir = "."):

    data = read_pickled_data(data_pickle)
    params = json.load(open(parameter_space))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    run_models(params,
               data["train_images"],
               data["train_labels"],
               data["valid_images"],
               data["valid_labels"],
               output_dir)


if __name__ == "__main__":
    argh.dispatch_command(main)
