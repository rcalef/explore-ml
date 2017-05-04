#! /usr/bin/env python3

import argh
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

from explore_ml.utils.data_utils import *
from explore_ml.utils.image_utils import *


def get_preprocessed_image(image_fn,kernel_size = 5):
    grey = load_to_greyscale(image_fn)
    return preprocess_image(grey, kernel_size)

def get_filtered_bounds(binarized):
    contour_bounds = get_contour_boxes(binarized)
    return median_height_filter(contour_bounds)

def resize_and_clean(subimg,erode_iterations = 5):
    res = cv2.resize(subimg,(28,28))
    cleaned = cv2.erode(res,np.ones((1,1)),erode_iterations)
    return cleaned

def predict_and_annotate(model,subimgs,boxes,original):
    annotated = original.copy()
    for (x,y,w,h),subimg in zip(boxes,subimgs):
        norm = normalize_image(subimg) 
        to_predict = resize_and_clean(norm)
        predicted = model.predict(to_predict.reshape(1,-1))

        cv2.rectangle(annotated,(x,y),(x+w,y+h),(255,255,255),10)
        cv2.putText(annotated,
                    str(int(predicted[0])),
                    (x,y+int(h*1.5)),
                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                    4,
                    (255,255,255),
                    5,
                    lineType=cv2.LINE_AA)
    return annotated

        

    
@argh.arg("-i","--image",required=True,help="Input image to classify "
          "digits in.")
@argh.arg("-m","--model",required=True,help = "Model dump "
          "from sklearn.joblib.")
@argh.arg("-k","--kernel-size",help = "Kernel size for smoothing image.")
@argh.arg("-p","--padding",help = "Fraction of bounding rectangle width "
          "and height to add when extracting contour sub-images.")
def main(image = None,
         model = None,
         kernel_size = 5,
         padding = 0.5,
         output = "classified.png"):

    classifier = joblib.load(model)
    smooth_image = get_preprocessed_image(image,kernel_size)
    binarized = binarize(smooth_image)

    filtered_bounds = get_filtered_bounds(binarized)
    filt_subimgs = extract_contour_subimgs(smooth_image,
                                           filtered_bounds,
                                           padding)

    annotated_img = predict_and_annotate(classifier,
                                         filt_subimgs,
                                         filtered_bounds,
                                         binarized)

    plt.imshow(annotated_img)
    plt.savefig(output)


if __name__ == "__main__":
    argh.dispatch_command(main)
