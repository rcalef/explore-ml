import cv2
import numpy as np

def load_to_greyscale(image_fn):
    return cv2.imread(image_fn,flags=0)

def normalize_image(img,pixel_depth = 255):
    return (img - (pixel_depth / 2)) / pixel_depth

def preprocess_image(img,ksize):
    opened = cv2.morphologyEx(img,cv2.MORPH_OPEN,np.ones((ksize,ksize)))
    smooth = cv2.GaussianBlur(opened,(ksize,ksize),0)
    return smooth

def binarize(img):
    thresh,binarized = cv2.threshold(img,
                                     0,
                                     255,
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binarized

def get_contour_boxes(binarized):
    dest,contours,hierarchy = cv2.findContours(binarized.copy(),
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
    return [cv2.boundingRect(contour) for contour in contours]

def extract_contour_subimgs(img,bounds,pad = 0.2):
    subimgs = []
    lower_pad = 1 + pad

    for x,y,w,h in bounds:
        xmin = max(0, x - int(w * pad))
        xmax = x + int(w * lower_pad)
        ymin = max(0, y - int(h * pad))
        ymax = y + int(h * lower_pad)

        subimgs.append(img[ymin:ymax,xmin:xmax])
    return subimgs

def median_area_filter(boxes,thresh = 3.8):
    #Get the median box area
    median_area = np.median([w * h for (x,y,w,h) in boxes])
    lb =  median_area / thresh
    ub =  median_area * thresh

    #Only keep sub images with area within an order of magnitude of the median
    return [(x,y,w,h) for (x,y,w,h) in boxes \
            if  w * h > lb \
            and w * h < ub]

def median_height_filter(boxes,thresh = 1.2):
    #Get the median box height
    median_height = np.median([h for (x,y,w,h) in boxes])
    lb =  median_height / thresh
    ub =  median_height * thresh

    #Only keep sub images with area within an order of magnitude of the median
    return [(x,y,w,h) for (x,y,w,h) in boxes \
            if  h > lb \
            and h < ub]


