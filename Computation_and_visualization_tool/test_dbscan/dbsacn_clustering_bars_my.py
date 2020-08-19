import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

data_tensors = pd.read_csv("/Computation-and-visualization-tool/Tensor_field_computation/tensor_vote_matrix.csv", sep=",", index_col=False)
X = len(data_tensors["X"].unique())
Y = len(data_tensors["Y"].unique())
cord_list = {
        "x_val": [],
        "y_val": [],
        "CL": []
    }

for i in range(X * Y):
     if (data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0) and (data_tensors["CL"][i] <= 0.05):
     # if data_tensors['CL'][i] > 0.0 and data_tensors['CL'][i] <= 0.05:
           cord_list["x_val"].append(data_tensors["X"][i])
           cord_list["y_val"].append(data_tensors["Y"][i])
           cord_list["CL"].append((data_tensors["CL"][i]))

plt.scatter(cord_list['x_val'], cord_list['y_val'], s=5)
plt.show()
data = np.array([cord_list['x_val'],cord_list['y_val']]).T
# cord_list1 = np.array(data).T
# Load data in X
db = DBSCAN(eps=3, min_samples=4).fit(data)
labels = db.labels_
centers=[]
cord_list_y=[]
cord_list_x=[]
for i in (np.unique(labels)[1:]):
    indexes = [id for id in range(len(labels)) if labels[id] == i]
    x=0
    y=0
    for k in indexes:
        x+=cord_list['x_val'][k]
        y+=cord_list['y_val'][k]
        cord_list_y.append(y)
        cord_list_x.append(x)
    centers+=[[x//len(indexes),y//len(indexes)]]
centers=sorted(centers)

df1= pd.DataFrame(cord_list['x_val'],columns=["x"])
df2= pd.DataFrame(cord_list['y_val'], columns=["y"])
df3= pd.DataFrame(db.labels_ ,columns=["label"])
df = pd.concat([df1,df2,df3], axis=1)
# plot data with seaborn
facet = sns.lmplot(data=df, x='x', y='y', hue='label',
                   fit_reg=False, legend=True, legend_out=True)
plt.suptitle("Clustering points with black dots as cluster centers")
plt.plot(np.array(centers)[:,0], np.array(centers)[:,1], 'ko')
plt.show()
# my code

points = centers
cords_xy = sorted(points, key = lambda x: x[0])
cords_xy1 = sorted(points, key = lambda x: x[1])
print('p',points)
print('xy1',cords_xy1)

min_y = min(cord_list_y)
print ("y_min:", min_y)
diff_x = []
cord_list_x = np.unique(np.array(sorted(cord_list_x)))

final_cord_x =[]
final_cord_y =[]
cords_len = len(cords_xy1)
final_list = []
for pts in cords_xy:
    if pts[1] > (min_y + 5) and pts not in final_list:
        final_list.append(pts)
cords_len = len(final_list)
# print(cords_len)
y = final_list[0][1]
i=0
# for i in range(cords_len-1):
while i < cords_len-1:
    if final_list[i][1] == final_list[i+1][1]:
        temp_x = (final_list[i][0] + final_list[i+1][0])/2
        temp_y = (final_list[i][1] + final_list[i+1][1])/2
        final_cord_x.append(temp_x)
        final_cord_y.append(temp_y)
        i += 2

    else:
        i += 1
# print("final", final_list, final_cord_x, final_cord_y)
# print (min_y)
for i in range(len(cords_xy)-1):
    if cords_xy[i][1] >= (min_y + 5):
        # print cords_xy[i][0]
        diff = cords_xy[i+1][0] - cords_xy[i][0]
        diff_x.append(diff)

# plt.hist(diff_x)
# plt.show()
L=[]
base_y=cords_xy1[0][1]
equal_y=[list(item[1]) for item in itertools.groupby(cords_xy1, key=lambda x:x[1])]
for i in equal_y:
    if len(i)==2:
        bar_width = i[1][0]-i[0][0]
        pixellen_y = i[0][1]-base_y
        pixelcenter_x = (i[1][0]+i[0][0])//2
        L.append([pixelcenter_x,pixellen_y,bar_width])
# sort based on X centers
L = np.array(sorted(L, key = lambda x: x[0]))
print("L:",len(L))
pixelcenter_x = L[:,0]
pixellen_y = L[:,1]
bar_width= L[:,2]
# width = 30
# width = 40
fi, ax= plt.subplots()
ax.axis('on')
# for point in final_list:
#     plt.scatter(point[0], point[1], color='b', s=5)
# plt.scatter(final_cord_x, final_cord_y, s=5, color='k')
for i in range(len(final_cord_y)):
    plt.bar(final_cord_x[i], final_cord_y[i],10, color=[(0.2, 0.2, 0.2)])

plt.show()
# df = pd.DataFrame(np.array(final_cord_x), columns=['X'])
# df1 = pd.DataFrame(np.array(final_cord_y), columns=['Y'])
# df_final = pd.concat([df, df1], axis=1)
# df_final.to_csv("reconstructed_data.csv", index=False, sep=",")
# xy1 = np.array(list(map(list, zip(final_cord_x, final_cord_y))))
# data = pd.read_csv("/home/komaldadhich/Documents/study/Research/graph_percept/source/project/chart_percept/charts/final-charts/bars/bc04/biostats.csv",  sep=",", index_col=False)
# # xy2 = np.array(list(map(list, zip(data[' "Age"'], data[' "Weight (lbs)"']))))
# # xy2 = np.array(list(map(list, zip(data['X'], data['Y']))))
# # emd_calculation(xy2, xy1)
#
# # print (len(cord_list_y))
# ymin = min(final_cord_y)
# ymax = max(final_cord_y)
# # print (ymax, ymin)
# org_list = sorted(data[' "Weight (lbs)"'].tolist())
# print (org_list)
# ymin_org = min(org_list)
# ymax_org = max(org_list)
#
# norm_y = [(float(i-ymin)/float(ymax-ymin)) for i in sorted(final_cord_y)]
# norm_y_org = [(float(j-ymin_org)/float(ymax_org-ymin_org)) for j in org_list]
#
# # norm_y = np.histogram(norm_y)
# # norm_y_org = np.histogram(norm_y_org)
# print ("Reconstructed y:", norm_y)
# print ("Original y:", norm_y_org)
#
# # plt.bar(pixelcenter_x,height=pixellen_y, width=bar_width)
# # plt.bar(pixelcenter_x,height=pixellen_y, width=bar_width, bottom=base_y)
# plt.show()
#
# # print wasserstein_distance(norm_y, norm_y_org)