import numpy as np

def compute_features(eigen_val, e_val_sum):
    # e_val_sum = eigen_val[0] + eigen_val[1]
    # cp = 2*eigen_val[1]/e_val_sum
    diff = eigen_val[0] - eigen_val[1]
    cl = (diff) / e_val_sum
    # cl = 1 - cp
    cp = 1 - cl
    return cl, cp

def compute_saliency(e_val1, e_val2):
    val_sum = e_val1 + e_val2
    w = [e_val1, e_val2]
    cl, cp = compute_features(w, val_sum)

    return cl, cp
