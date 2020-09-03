import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parameterised_space(x_cords, y_cords, x_max, y_max):
    x_cords = x_cords/x_max
    y_cords = y_cords/y_max
    return x_cords, y_cords

data_tensors = pd.read_csv("/home/komaldadhich/Documents/study/Research/graph_percept/source/project/chart_percept/Computation-and-visualization-tool/Tensor_field_computation/tensor_vote_matrix.csv", sep=",", index_col=False)
X = data_tensors['X']
Y = data_tensors['Y']
cl = data_tensors['CL']

x_max = np.max(X)
y_max = np.max(Y)

print x_max, y_max

x_co, y_co = parameterised_space(X, Y, x_max, y_max)

cord_list = {
        "x_val": [],
        "y_val": [],
        "CL": []
    }

cord_list1 = {
        "x_val": [],
        "y_val": [],
        "CL": []
    }

for i in range(len(X)):
    if data_tensors['CL'][i]>0.0 and data_tensors['CL'][i]<=0.1:
        cord_list["x_val"].append(data_tensors["X"][i])
        cord_list["y_val"].append(data_tensors["Y"][i])

# for i in range(len(X)):
#     if data_tensors['CL'][i]>0.15 and data_tensors['CL'][i]<=0.2:
#         cord_list1["x_val"].append(data_tensors["X"][i])
#         cord_list1["y_val"].append(data_tensors["Y"][i])

plt.scatter(cord_list['x_val'], cord_list['y_val'], s= 2)
# plt.scatter(cord_list1['x_val'], cord_list1['y_val'], s= 2, color='black')
plt.show()