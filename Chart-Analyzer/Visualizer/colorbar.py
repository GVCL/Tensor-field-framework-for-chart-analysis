import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

csv_file_path = "/home/komaldadhich/Desktop/test_results/bars/bars/bc01/tensor_vote_matrix.csv"
data_tensors = pd.read_csv(csv_file_path, sep=",", index_col=False)
X = len(data_tensors["X"].unique())
Y = len(data_tensors["Y"].unique())

cord_list = {
    "x_val": [],
    "y_val": [],
    "e1_vec": [],
    "e2_vec": [],
    "e3_vec": [],
    "e4_vec": [],
    "ani_val1": [],
    "ani_val2": [],
    "val1": [],
    "val2": [],
    "CL": [],
    "CP": []
}

for i in range(X * Y):

    if data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0:
        cord_list["x_val"].append(data_tensors["X"][i])
        cord_list["y_val"].append(data_tensors["Y"][i])
        cord_list["ani_val1"].append((data_tensors["ani_val1"][i]))
        cord_list["ani_val2"].append((data_tensors["ani_val2"][i]))
        cord_list["val1"].append((data_tensors["val1"][i]))
        cord_list["val2"].append((data_tensors["val2"][i]))
        cord_list["e1_vec"].append(data_tensors["ani_val2"][i] * data_tensors["vec00"][i])
        cord_list["e2_vec"].append(data_tensors["ani_val2"][i] * data_tensors["vec01"][i])
        cord_list["e3_vec"].append(data_tensors["ani_val1"][i] * data_tensors["vec10"][i])
        cord_list["e4_vec"].append(data_tensors["ani_val1"][i] * data_tensors["vec11"][i])
        cord_list["CL"].append((data_tensors["CL"][i]))
        cord_list["CP"].append((data_tensors["CP"][i]))

print("Visualizing Tensor Voting")
xy_cord = zip(cord_list["x_val"], cord_list["y_val"])
ells=[]
count =0

c_map = plt.cm.get_cmap('coolwarm', 100)(np.array(cord_list["CL"]))
c_map1 = plt.cm.get_cmap('coolwarm', 100)
print(c_map1(0))
print(c_map1(0.5))
print(c_map1(1))


# print(cm.coolwarm(0))