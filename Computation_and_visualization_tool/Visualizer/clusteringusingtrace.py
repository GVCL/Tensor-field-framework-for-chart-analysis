from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from  colormap import *
from sklearn.cluster import *
from operator import add
from skimage import color
from PIL import Image
import seaborn as sns
import cv2

data_tensors = pd.read_csv("../Tensor_field_computation/tensor_vote_matrix_bc01.csv", sep=",", index_col=False)
# data_tensors['edge'] = edges[::-1].reshape(-1,1)
X = len(data_tensors["X"].unique())
Y = len(data_tensors["Y"].unique())
print(X,Y)
cord_list = {
        "x_val": [],
        "y_val": [],
        "CL": []
    }

# eigen trace computation normalized between 0 to 1
trace=list(map(add,  data_tensors["val1"], data_tensors["val2"]))
amin, amax = min(trace), max(trace)
for i, val in enumerate(trace):
    trace[i] = (val-amin) / (amax-amin)

for i in range(X * Y):
     if (data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0) and (data_tensors["CL"][i] < 0.5) and (trace[i]>0.01):
          # if(data_tensors["X"][i]<(X-5) and data_tensors["X"][i]>5 and data_tensors["Y"][i]>5 and data_tensors["Y"][i]<(Y-5)) :
           cord_list["x_val"].append(data_tensors["X"][i])
           cord_list["y_val"].append(data_tensors["Y"][i])
           cord_list["CL"].append((data_tensors["CL"][i]))
c_map = plt.cm.get_cmap('coolwarm', 100)(np.array(cord_list["CL"]))
fig, ax = plt.subplots()
plt.suptitle("Saliency Visualization")
ax.axis('on')
Q1 = plt.scatter(cord_list["x_val"], cord_list["y_val"], s=1, c=cord_list['CL'], cmap=plt.cm.get_cmap('coolwarm', 100))
plt.colorbar(Q1)

data = np.array([cord_list['x_val'],cord_list['y_val']]).T

db = DBSCAN(eps=7, min_samples=5).fit(data)
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
print(np.array(centers))

df1= pd.DataFrame(cord_list['x_val'],columns=["x"])
df2= pd.DataFrame(cord_list['y_val'], columns=["y"])
df3= pd.DataFrame(db.labels_ ,columns=["label"])
df = pd.concat([df1,df2,df3], axis=1)
#plot data with seaborn
facet = sns.lmplot(data=df, x='x', y='y', hue='label',
                   fit_reg=False, legend=True, legend_out=True)
plt.suptitle("Clustering points with black dots as cluster centers")
plt.plot(np.array(centers)[:,0], np.array(centers)[:,1], 'ko')

plt.show()




