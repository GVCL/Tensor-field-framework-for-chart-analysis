import pandas as pd
from compute_saliency import *
from scale import *
import numpy as np

def generate_tensor_vote(data):
    location_data = data[["X", "Y"]].copy()
    X = len(data["X"].unique())
    Y = len(data["Y"].unique())
    print (X, Y)
    xx = np.array(data["xx"]).reshape(Y, X)
    xy = np.array(data["xy"]).reshape(Y, X)
    yx = np.array(data["yx"]).reshape(Y, X)
    yy = np.array(data["yy"]).reshape(Y, X)

    tv = [[0 for i in range(X)] for j in range(Y)]
    e_val1 = [[0 for i in range(X)] for j in range(Y)]
    e_val2 = [[0 for i in range(X)] for j in range(Y)]
    e_vec00 = [[0 for i in range(X)] for j in range(Y)]
    e_vec01 = [[0 for i in range(X)] for j in range(Y)]
    e_vec10 = [[0 for i in range(X)] for j in range(Y)]
    e_vec11 = [[0 for i in range(X)] for j in range(Y)]
    cl_cp = [[[] for i in range(X)] for j in range(Y)]
    val1_anisotropic = [[[] for i in range(X)] for j in range(Y)]
    scale_val = [[0 for i in range(X)] for j in range(Y)]
    entropy = [[0 for i in range(X)] for j in range(Y)]
    vector_major = [[[] for i in range(X)] for j in range(Y)]
    vector_minor = [[[] for i in range(X)] for j in range(Y)]
    v1 = np.zeros((Y, X))
    v2 = np.zeros((Y, X))
    for row in range(Y):
        # tv.append([])
        # e_val1.append([])
        # e_vec1.append([])
        # cl_cp.append([])
        # scale_val.append([])
        # val1_anisotropic.append([])
        # entropy.append([])
        # vector.append([])
        for col in range(X):
            tensor_vote = [[0 for i in range(2)] for j in range(2)]
            cl, cp = 0, 0

            E_min = 0
            optimal_scale = 0
            a_val1, a_val2 = 0, 0
            e1 = [0, 0]
            e2 = [0, 0]
            for scale in range(1, 10):
                print (row, col)
                S_ij = [[0 for i in range(2)] for j in range(2)]
                neighbour_list = find_neighbour_new(row, col, X, Y, scale)

                sigma = len(neighbour_list)
                K_j = [[0 for i in range(2)] for j in range(2)]
                K1 = [[1, 0],[0, 0]]
                K3 = [[1, 0],[0, 1]]

                for n in neighbour_list:
                    c = np.exp(-(np.sqrt(np.square(row-n[0]) + np.square(col-n[1])))/sigma)
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
                    # temp_matrix = (K1 * R2)
                    # temp_matrix2 = (K3 * R2)
                    temp_tensor = c * R * temp_matrix
                    # temp_tensor2 = c * R * temp_matrix2
                    # tensor_vote = S_ij
                    S_ij = np.add(S_ij, temp_tensor)
                    # S_ij = np.add(S_ij, temp_tensor2)
                    # tensor_vote = np.add(S_ij, tensor_vote)

                    # tv[row].append(S_ij)
                w, v = np.linalg.eig(S_ij)

                if w[0] >= w[1]:
                    t = 0.16
                    if w[1] < 0:
                        w[1] = 0
                    if w[0] < 0:
                        w[0] = 0

                    # val1_anisotropic[row].append(np.exp(-w[::-1] / t))
                    val1 = np.exp(-w[1]/t)
                    val2 = np.exp(-w[0]/t)
                    e_val1[row][col] = w[0]
                    e_val2[row][col] = w[1]
                    e_vec00[row][col] = v[0][0]
                    e_vec01[row][col] = v[0][1]
                    e_vec10[row][col] = v[1][0]
                    e_vec11[row][col] = v[1][1]

                    e_val_sum = e_val1[row][col] + e_val1[row][col]
                    if (e_val_sum != 0):
                        cl_val, cp_val = compute_saliency(val1, val2)
                        # cl_val, cp_val = compute_saliency(row, col, val1_anisotropic[row][col][0], val1_anisotropic[row][col][1])

                    else:
                        cl_val, cp_val = 0, 0

                    # cl_cp[row].append([cl_val, cp_val])

                else:
                    t = 0.16
                    if w[0] < 0:
                        w[0] = 0
                    if w[1] < 0:
                        w[1] = 0
                    # val1_anisotropic[row].append(np.exp(-w / t))
                    e_val1[row][col] = w[1]
                    e_val2[row][col] = w[0]

                    e_vec00[row][col] = v[1][0]
                    e_vec01[row][col] = v[1][1]
                    e_vec10[row][col] = v[0][0]
                    e_vec11[row][col] = v[0][1]

                    val1 = np.exp(-w[0]/t)
                    val2 = np.exp(-w[1]/t)

                    e_val_sum = e_val1[row][col] + e_val1[row][col]
                    if (e_val_sum != 0):
                        cl_val, cp_val = compute_saliency(val1, val2)
                        # cl_val, cp_val = compute_saliency(row, col, val1_anisotropic[row][col][0], val1_anisotropic[row][col][1])
                    else:
                        cl_val, cp_val = 0, 0

                    # cl_cp[row].append([cl_val, cp_val])
                    # print cl_val, cp_val
                if (cl_val != 0 and cp_val != 0):
                    shannon_entropy = -cl_val * np.log(cl_val) - cp_val * np.log(cp_val)
                else:
                    shannon_entropy = 0
                # sigma_var = [math.sqrt(val1), math.sqrt(val2)]
                # diff_sigma = sigma_var[0] - sigma_var[1]
                # sum_sigma = sigma_var[0] + sigma_var[1]
                # if sum_sigma == 0:
                #     shannon_entropy = np.nan
                # #   entropy[row].append(shannon_entropy)
                # else:
                #     feature1 = diff_sigma / sum_sigma
                #     feature2 = sigma_var[1] / sum_sigma
                #     shannon_entropy = -feature1 * np.log(feature1) - feature2 * np.log(feature2)
                # tensor_vote = S_ij
                if ((E_min > shannon_entropy) or (scale == 1)):
                    # print "In Condition"

                    E_min = shannon_entropy
                    optimal_scale = scale
                    tensor_vote = S_ij
                    cl, cp = cl_val, cp_val
                    a_val1, a_val2 = val1, val2
                    e1 = [e_vec00[row][col], e_vec01[row][col]]
                    e2 = [e_vec10[row][col], e_vec11[row][col]]

                # else:
                #     E_min = E_min
                #     optimal_scale = optimal_scale
                #     tensor_vote = tensor_vote


                # entropy[row].append(E_min)
                # scale_val[row].append(optimal_scale)
                # tv[row].append(tensor_vote)

            # scale_val[row].append(optimal_scale)
            # val1_anisotropic[row].append([a_val1, a_val2])
            # cl_cp[row].append([cl, cp])
            # entropy[row].append(E_min)
            # tv[row].append(tensor_vote)
            # vector[row].append([e1, e2])

            scale_val[row][col] = optimal_scale
            val1_anisotropic[row][col] = [a_val1, a_val2]
            cl_cp[row][col] = [cl, cp]
            entropy[row][col] = E_min
            tv[row][col] = tensor_vote
            # vector_major[row][col] = e1
            # vector_minor[row][col] = e2
            v1[row][col] = e2[0]
            v2[row][col] = e2[1]

    cl_cp = np.array(cl_cp)
    e_vector1 = np.array(v1)
    e_vector2 = np.array(v2)
    cl_cp_value = cl_cp.reshape(Y*X, 2)
    scale_val = np.array(scale_val).reshape(Y*X, 1)
    entropy_mat = np.array(entropy).reshape(Y*X, 1)
    val1_anisotropic_array = np.array(val1_anisotropic)
    anisotropic_val_mat = val1_anisotropic_array.reshape(Y * X, 2)
    eigen_vec_mat1 = e_vector1.reshape(Y * X, 1)
    eigen_vec_mat2 = e_vector2.reshape(Y * X, 1)
    df_eigen_vec = pd.DataFrame(
        eigen_vec_mat1, columns=["vec00"]
    )
    df_eigen_vec2 = pd.DataFrame(
        eigen_vec_mat2, columns=["vec01"]
    )
    df_cl_cp_val = pd.DataFrame(cl_cp_value, columns=["CL", "CP"])
    df_aniso_val = pd.DataFrame(anisotropic_val_mat, columns=["ani_val1", "ani_val2"])
    df_variance = pd.DataFrame(scale_val, columns=["optimal scale"])
    df_entropy = pd.DataFrame(entropy_mat, columns=["Entropy"])
    print ("concatination")
    df = pd.concat([location_data, df_eigen_vec,df_eigen_vec2, df_aniso_val, df_variance, df_entropy, df_cl_cp_val], axis=1)
    # df = pd.concat([location_data, df_eigen_val, df_eigen_vec,df_aniso_val, df_variance, df_entropy, df_cl_cp_val], axis=1)
    # df = pd.concat([location_data, df_eigen_val, df_eigen_vec, df_aniso_val, df_cl_cp_val], axis=1)
    # df = df.replace(np.nan, '', regex=True)
    print ("writing in csv")
    df.to_csv("tensor_vote_matrix_optimal_hist_poss_x_1_10.csv", index=False, sep=",")
    # df1 = pd.read_csv("tensor_vote_matrix_optimal.csv", sep=",", index_col=False)
    # data_vote = pd.concat([location_data, df1]).groupby(level=0).mean()
    # print list(data_vote.columns.values)
    print("Tensor Voting computation done!")
    # return data_vote
    # visualize_tensor_voting(df) # to be removed




