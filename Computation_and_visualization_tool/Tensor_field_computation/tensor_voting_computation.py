import pandas as pd
from compute_saliency import *
from scale import *
import os

def generate_tensor_vote(data, filename):
    image_name = os.path.basename(filename).split(".png")[0]
    path = os.path.dirname(filename)+'/'
    location_data = data[["X", "Y"]].copy()
    X = len(data["X"].unique())
    Y = len(data["Y"].unique())

    xx = np.array(data["xx"]).reshape(Y, X)
    xy = np.array(data["xy"]).reshape(Y, X)
    yx = np.array(data["yx"]).reshape(Y, X)
    yy = np.array(data["yy"]).reshape(Y, X)

    tv = []
    e_val1 = []
    e_vec1 = []
    cl_cp = []
    val1_anisotropic = []
    sigma_var = []
    entropy = []
    for row in range(Y):
        tv.append([])
        e_val1.append([])
        e_vec1.append([])
        cl_cp.append([])
        sigma_var.append([])
        val1_anisotropic.append([])
        entropy.append([])
        for col in range(X):
            neighbour_list = find_neighbour(row, col, X, Y, 1)

            scale = len(neighbour_list)
            S_ij = [[0 for i in range(2)] for j in range(2)]
            K_j = [[0 for i in range(2)] for j in range(2)]
            K1 = [[1, 0],[0, 0]]
            K3 = [[1, 0],[0, 1]]
            for n in neighbour_list:
                c = np.exp(-(np.sqrt(np.square(row-n[0]) + np.square(col-n[1])))/scale)
                r_ij = np.array([row - n[0], col - n[1]]).reshape(2, 1)
                temp = 2 * (r_ij * r_ij.T)
                R = np.identity(2) - temp
                temp = 0.5 * (r_ij * r_ij.T)
                temp2 = np.identity(2) - temp
                R2 = temp2 * R

                K_j[0][0] = xx[n[0]][n[1]]
                K_j[0][1] = xy[n[0]][n[1]]
                K_j[1][0] = yx[n[0]][n[1]]
                K_j[1][1] = yy[n[0]][n[1]]


                temp_matrix = (K_j * R2)
                temp_tensor = c * R * temp_matrix
                S_ij = np.add(S_ij, temp_tensor)

            tv[row].append(S_ij)
            w, v = np.linalg.eig(S_ij)

            if w[0] >= w[1]:

                t = 0.16
                if w[1] < 0:
                    w[1] = 0
                if w[0] < 0:
                    w[1] = 0

                val1_anisotropic[row].append(np.exp(-w[::-1] / t))
                e_val1[row].append(w)

                e_vec1[row].append(v)
                # e_val_sum = e_val1[row][col][0] + e_val1[row][col][1]
                e_val_sum = e_val1[row][col][0] + e_val1[row][col][1]
                if (e_val_sum != 0):
                    cl_val, cp_val = compute_saliency(val1_anisotropic[row][col][0],
                                                      val1_anisotropic[row][col][1])
                else:

                    cl_val, cp_val = 0, 0
                cl_cp[row].append([cl_val, cp_val])
            else:
                t = 0.16
                if w[0] < 0:
                    w[0] = 0
                val1_anisotropic[row].append(np.exp(-w / t))
                e_val1[row].append(w[::-1])
                e_vec1[row].append(v[::-1])

                e_val_sum = e_val1[row][col][0] + e_val1[row][col][1]
                if (e_val_sum != 0):
                    cl_val, cp_val = compute_saliency(val1_anisotropic[row][col][0], val1_anisotropic[row][col][1])
                else:

                    cl_val, cp_val = 0, 0
                cl_cp[row].append([cl_val, cp_val])

    e_val1 = np.array(e_val1)
    cl_cp = np.array(cl_cp)
    val1_anisotropic_array = np.array(val1_anisotropic)
    cl_cp_value = cl_cp.reshape(Y * X, 2)
    anisotropic_val_mat = val1_anisotropic_array.reshape(Y * X, 2)
    e_vec1 = np.asarray(e_vec1)
    eigen_val_mat = e_val1.reshape(Y * X, 2)
    eigen_vec_mat = e_vec1.reshape(Y * X, 4)
    df_eigen_vec = pd.DataFrame(
        eigen_vec_mat, columns=["vec00", "vec01", "vec10", "vec11"]
    )
    df_eigen_val = pd.DataFrame(eigen_val_mat, columns=["val1", "val2"])
    df_cl_cp_val = pd.DataFrame(cl_cp_value, columns=["CL", "CP"])
    df_aniso_val = pd.DataFrame(anisotropic_val_mat, columns=["ani_val1", "ani_val2"])
    df = pd.concat([location_data, df_eigen_val, df_eigen_vec, df_aniso_val, df_cl_cp_val], axis=1)

    df.to_csv(path+"tensor_vote_matrix_"+image_name+".csv", index=False, sep=",")
    print("Tensor Voting computation done!")







