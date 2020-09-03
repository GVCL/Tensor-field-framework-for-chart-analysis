from scipy import ndimage as ndi
import cv2
import os
import pandas as pd
import numpy as np
from compute_saliency import *
from tensor_voting_computation import generate_tensor_vote


def color_normalization(arr, arr1, arr2, X, Y):
    # Normalizing LAB values
    l_max = 100
    l_min = 0
    a_max = 127
    a_min = -127
    b_max = 127
    b_min = -127
    l_norm = []
    a_norm = []
    b_norm = []
    for p in range(Y):
        l_norm.append([])
        a_norm.append([])
        b_norm.append([])
        for q in range(X):
            x = (arr[p][q] - l_min) / (l_max - l_min)
            y = (arr1[p][q] - a_min) / (a_max - a_min)
            z = (arr2[p][q] - b_min) / (b_max - b_min)
            l_norm[p].append(x)
            a_norm[p].append(y)
            b_norm[p].append(z)

    return l_norm, a_norm, b_norm


def _compute_derivatives(image, mode='constant', cval=0):
    imy = ndi.sobel(image, axis=0, mode=mode, cval=cval)
    imx = ndi.sobel(image, axis=1, mode=mode, cval=cval)

    return imx, imy

def structure_tensor(image, sigma=1, mode='constant', cval=0):
    imx, imy = _compute_derivatives(image, mode=mode, cval=cval)

    IMxx = imx * imx
    IMxy = imx * imy
    IMyy = imy * imy
    # structure tensor
    Axx = cv2.GaussianBlur(IMxx,(1,1), sigma)
    Axy = cv2.GaussianBlur(IMxy,(1,1), sigma)
    Ayy = cv2.GaussianBlur(IMyy,(1,1), sigma)

    return Axx, Axy, Ayy


def compute_structure_tensor(data, image, filename):
    image_name = os.path.basename(filename).split(".png")[0]
    path = os.path.dirname(filename)+'/'
    location_data = data[["X", "Y"]].copy()
    X = len(data["X"].unique())
    Y = len(data["Y"].unique())

    arr = np.array(image[:, :, 0])
    arr1 = np.array(image[:, :, 1])
    arr2 = np.array(image[:, :, 2])

    l_norm, a_norm, b_norm = color_normalization(arr, arr1, arr2, X, Y)
    Rxx, Rxy, Ryy = structure_tensor(l_norm, sigma=0.1)
    Bxx, Bxy, Byy = structure_tensor(a_norm, sigma=0.1)
    Gxx, Gxy, Gyy = structure_tensor(b_norm, sigma=0.1)
    Fxx = []
    Fxy = []
    Fyy = []
    cov = []
    eigen_val = []
    eigen_vec = []
    cl_cp_list = []
    count = 0
    for i in range(Y):
        eigen_val.append([])
        eigen_vec.append([])
        cov.append([])
        for j in range(X):
            temp_xx = Rxx[i][j] + Bxx[i][j] + Gxx[i][j]
            temp_xy = Rxy[i][j] + Bxy[i][j] + Gxy[i][j]
            temp_yy = Ryy[i][j] + Byy[i][j] + Gyy[i][j]


            w, v = np.linalg.eig([[temp_xx, temp_xy], [temp_xy, temp_yy]])
            cov[i].append([[temp_xx, temp_xy], [temp_xy, temp_yy]])

            if w[0] >= w[1]:
                eigen_val[i].append(w)
                eigen_vec[i].append(v)
                if (eigen_val[i][j][1] < 0):
                    eigen_val[i][j][1] = 0
                e_val_sum = eigen_val[i][j][0] + eigen_val[i][j][1]
                if (e_val_sum != 0):
                    cl_val, cp_val = compute_saliency(eigen_val[i][j][0], eigen_val[i][j][1])
                else:
                    cl_val, cp_val = 0, 0
            else:
                eigen_val[i].append(w[::-1])
                eigen_vec[i].append(v[::-1])
                if (eigen_val[i][j][1] < 0):
                    eigen_val[i][j][1] = 0
                e_val_sum = eigen_val[i][j][0] + eigen_val[i][j][1]
                if (e_val_sum != 0):
                    cl_val, cp_val = compute_saliency(eigen_val[i][j][0], eigen_val[i][j][1])
                else:
                    cl_val, cp_val = 0, 0

            cl_cp_list.append([cl_val, cp_val])

    cov_mat = np.array(cov)

    eigen_val = np.array(eigen_val)
    eigen_vec = np.array(eigen_vec)
    cl_cp_value = np.array(cl_cp_list)
    cl_cp_value = cl_cp_value.reshape(Y * X, 2)

    cov_mat = cov_mat.reshape(Y * X, 4)
    eigen_val_mat = eigen_val.reshape(Y * X, 2)
    eigen_vec_mat = eigen_vec.reshape(Y * X, 4)

    df_cl_cp_val = pd.DataFrame(cl_cp_value, columns=["CL", "CP"])
    df_cov_mat = pd.DataFrame(cov_mat, columns=["xx", "xy", "yx", "yy"])
    df_eigen_val = pd.DataFrame(eigen_val_mat, columns=["val1", "val2"])
    df_eigen_vec = pd.DataFrame(
        eigen_vec_mat, columns=["vec00", "vec01", "vec10", "vec11"]
    )

    df = pd.concat([location_data, df_cov_mat, df_eigen_val, df_eigen_vec, df_cl_cp_val], axis=1)
    df.to_csv(path+"structure_tensor_"+image_name+".csv", index=False, sep=",")
    print ("Computed Structure Tensor.")
    data_add = [[0 for i in range(2)] for j in range(Y*X)]
    # data_loc = [[0 for i in range(2)] for j in range(Y*X)]
    data_add = np.array(data_add, dtype=np.float64)

    # Tensor Voting computation
    generate_tensor_vote(df, filename)
