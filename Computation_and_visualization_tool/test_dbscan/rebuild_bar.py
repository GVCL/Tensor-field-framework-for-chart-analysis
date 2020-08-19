import pandas as pd
# from  Computation_visualization_tool.Visualizer.colormap import *
import seaborn as sns
from sklearn.cluster import *
import itertools
# import pytesseract
import cv2

data_tensors = pd.read_csv("/Computation-and-visualization-tool/Tensor_field_computation/tensor_vote_matrix.csv", sep=",", index_col=False)
X = len(data_tensors["X"].unique())
Y = len(data_tensors["Y"].unique())
print(X,Y)
cord_list = {
        "x_val": [],
        "y_val": [],
        "CL": []
    }

for i in range(X * Y):
     if (data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0) and (data_tensors["CL"][i] < 0.5):
         # if(data_tensors["X"][i]<(X-5) and data_tensors["X"][i]>5 and data_tensors["Y"][i]>5 and data_tensors["Y"][i]<(Y-5)) :
           cord_list["x_val"].append(data_tensors["X"][i])
           cord_list["y_val"].append(data_tensors["Y"][i])
           cord_list["CL"].append((data_tensors["CL"][i]))

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
centers=sorted(centers)

df1= pd.DataFrame(cord_list['x_val'],columns=["x"])
df2= pd.DataFrame(cord_list['y_val'], columns=["y"])
df3= pd.DataFrame(db.labels_ ,columns=["label"])
df = pd.concat([df1,df2,df3], axis=1)
#plot data with seaborn
facet = sns.lmplot(data=df, x='x', y='y', hue='label',
                   fit_reg=False, legend=True, legend_out=True)
plt.suptitle("Clustering points with black dots as cluster centers")
plt.plot(np.array(centers)[:,0], np.array(centers)[:,1], 'ko')


# bar plot
L=[]
centers = sorted(centers, key = lambda x: x[1])
base_y=centers[0][1]
equal_y=[list(item[1]) for item in itertools.groupby(centers, key=lambda x:x[1])]
for i in equal_y:
    if len(i)==2:
        bar_width = i[1][0]-i[0][0]
        pixellen_y = i[0][1]-base_y
        pixelcenter_x = (i[1][0]+i[0][0])//2
        L+=[[pixelcenter_x,pixellen_y,bar_width]]
# sort based on X centers
L = np.array(sorted(L, key = lambda x: x[0]))
pixelcenter_x = L[:,0]
pixellen_y = L[:,1]
bar_width= L[:,2]
plt.bar(pixelcenter_x,height=pixellen_y, width=bar_width, bottom=base_y)
plt.show()

