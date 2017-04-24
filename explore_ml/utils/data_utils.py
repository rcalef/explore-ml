#! /usr/bin/env python3

import pickle

import numpy as np

def read_int(fh,nbytes,order="big",signed=True):
    raw = fh.read(nbytes)
    return int.from_bytes(raw,byteorder=order,signed=signed)

def shuffle_dataset(data,labels):
    indices = np.random.permutation(len(data))
    return data[indices],labels[indices]

def make_validation_set(data,labels,nsamples=10000):
    new_data = data[:nsamples]
    new_labels = labels[:nsamples]
    valid_data = data[-nsamples:]
    valid_labels = labels[-nsamples:]
    return new_data,new_labels,valid_data,valid_labels

def write_pickled_data(tr_img,
                       tr_label,
                       val_img,
                       val_label,
                       ts_img,
                       ts_label,
                       output):
    data = {
        "train_images"  : tr_img,
        "train_labels" : tr_label,
        "valid_images"  : val_img,
        "valid_labels"  : val_label,
        "test_images"   : ts_img,
        "test_labels"   : ts_label
    }
    pickle.dump(data,open(output,"wb"))
