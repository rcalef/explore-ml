#! /usr/bin/env python3

import argh

from explore_ml.utils.mnist_utils import *
from explore_ml.utils.data_utils import *

def load_data(tr_images_fn,tr_labels_fn,te_images_fn,te_labels_fn,normalize):
    tr_images = read_mnist_images(tr_images_fn,normalize)
    tr_labels = read_mnist_labels(tr_labels_fn)

    te_images = read_mnist_images(te_images_fn,normalize)
    te_labels = read_mnist_labels(te_labels_fn)

    return tr_images,tr_labels,te_images,te_labels
    
@argh.arg("-tri","--training-images",required=True)
@argh.arg("-trl","--training-labels",required=True)
@argh.arg("-tei","--test-images",required=True)
@argh.arg("-tel","--test-labels",required=True)
def main(training_images= None,
         training_labels= None,
         test_images= None,
         test_labels= None,
         validation_size = 10000,
         normalize = True,
         output = "mnist.pickle"):

    # Read in the data from the raw gzip'd binary files
    tr_imgs,tr_lbls,te_imgs,te_lbls = load_data(training_images,
                                                training_labels,
                                                test_images,
                                                test_labels,
                                                normalize)

    # Shuffle the datasets
    shuf_tr_img,shuf_tr_lbl = shuffle_dataset(tr_imgs,tr_lbls)
    shuf_te_img,shuf_te_lbl = shuffle_dataset(te_imgs,te_lbls)

    # Make the validation set
    f_tr_img,f_tr_lbl,valid_img,valid_lbl = make_validation_set(shuf_tr_img,
                                                                shuf_tr_lbl,
                                                                validation_size)
    # Finally write it all out as a pickle
    write_pickled_data(f_tr_img,
                       f_tr_lbl,
                       valid_img,
                       valid_lbl,
                       shuf_te_img,
                       shuf_tr_lbl,
                       output)

if __name__ == "__main__":
    argh.dispatch_command(main)
