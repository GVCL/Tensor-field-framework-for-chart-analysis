from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# from  Computation_visualization_tool.Visualizer.colormap import *
from sklearn.cluster import *
from operator import add
from skimage import color
from PIL import Image
import itertools
import seaborn as sns
import cv2

data_tensors = pd.read_csv("/home/komaldadhich/Desktop/test_results/scatter/scatter/sc04/tensor_vote_matrix.csv", sep=",", index_col=False)
X = len(data_tensors["X"].unique())
Y = len(data_tensors["Y"].unique())
print(X,Y)
cord_list = {
        "x_val": [],
        "y_val": [],
        "CL": []
    }

a=list(map(add,  data_tensors["val1"], data_tensors["val2"]))
amin, amax = min(a), max(a)
for i, val in enumerate(a):
    a[i] = (val-amin) / (amax-amin)

for i in range(X * Y):
     # if (data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0) and (data_tensors["CL"][i] < 0.4) and (a[i]>0.01):
     if (data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0) and (data_tensors["CL"][i] < 0.4) and (a[i]>0.003):
         if(data_tensors["X"][i]<(X-5) and data_tensors["X"][i]>5 and data_tensors["Y"][i]>5 and data_tensors["Y"][i]<(Y-5)) :
             cord_list["x_val"].append(data_tensors["X"][i])
             cord_list["y_val"].append(data_tensors["Y"][i])
             cord_list["CL"].append((data_tensors["CL"][i]))

fig, ax = plt.subplots()
plt.suptitle("Critical points")
ax.axis('off')
Q1 = plt.scatter(cord_list["x_val"], cord_list["y_val"], s=1)
# plt.colorbar(Q1)
plt.show()

data = np.array([cord_list['x_val'],cord_list['y_val']]).T

# db = DBSCAN(eps=5, min_samples=3).fit(data) # correct and old
db = DBSCAN(eps=4, min_samples=3).fit(data) # correct and old

# db = DBSCAN(eps=10, min_samples=15).fit(data) #spade chart
labels = db.labels_

centers=[]
for i in (np.unique(labels)[1:]):
    indexes = [id for id in range(len(labels)) if labels[id] == i]
    x=0
    y=0
    for k in indexes:
        x+=cord_list['x_val'][k]
        y+=cord_list['y_val'][k]
    centers+=[[x//len(indexes),y//len(indexes)]]

# # To plot clusters with cluster centers
# df1= pd.DataFrame(cord_list['x_val'],columns=["x"])
# df2= pd.DataFrame(cord_list['y_val'], columns=["y"])
# df3= pd.DataFrame(db.labels_ ,columns=["label"])
# df = pd.concat([df1,df2,df3], axis=1)
# #plot data with seaborn
# facet = sns.lmplot(data=df, x='x', y='y', hue='label',
#                    fit_reg=False, legend=True, legend_out=True)
# plt.suptitle("Clustering points with black dots as cluster centers")
print(len(centers))

plt.plot(np.array(centers)[:,0], np.array(centers)[:,1], 'ko')
plt.axis('off')
plt.suptitle("Reconstructed Scatter")
plt.savefig('sc.png')

plt.show()
