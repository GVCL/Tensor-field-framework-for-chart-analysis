import pandas as pd
from  Computation_visualization_tool.Visualizer.colormap import *
import seaborn as sns
from sklearn.cluster import *
from operator import add
import itertools

data_tensors = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Topnvispaper/data/histogram/hg32/tensor_vote_matrix.csv", sep=",", index_col=False)
# data_tensors = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Computation_visualization_tool/Tensor_field_computation/tensor_vote_matrix.csv", sep=",", index_col=False)
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
     if (data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0) and (data_tensors["CL"][i] < 0.5) and (a[i]>0.003):
         cord_list["x_val"].append(data_tensors["X"][i])
         cord_list["y_val"].append(data_tensors["Y"][i])
         cord_list["CL"].append((data_tensors["CL"][i]))

data = np.array([cord_list['x_val'],cord_list['y_val']]).T

db = DBSCAN(eps=8, min_samples=2).fit(data)
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
centers = sorted(centers)
unused_centers = sorted(centers)

# df1= pd.DataFrame(cord_list['x_val'],columns=["x"])
# df2= pd.DataFrame(cord_list['y_val'], columns=["y"])
# df3= pd.DataFrame(db.labels_ ,columns=["label"])
# df = pd.concat([df1,df2,df3], axis=1)
# #plot data with seaborn
# facet = sns.lmplot(data=df, x='x', y='y', hue='label',
#                    fit_reg=False, legend=True, legend_out=True)
# plt.suptitle("Clustering points with black dots as cluster centers")
# plt.plot(np.array(centers)[:,0], np.array(centers)[:,1], 'ko')

# # histogram

def find_next(pt,flag,bin_centers,bin_height,bin_width,base_id):
    if pt in unused_centers:
        unused_centers.remove(pt)
    if(len(unused_centers)!=0):
        if(flag):
            # search in x-dirn having almost same y value
            y_vals = list(abs(np.array(unused_centers)[:,1] - pt[1]))
            dist = [[abs((unused_centers[i])[0] - pt[0])+y_vals[i] , i] for i in range(len(y_vals)) if y_vals[i]<neigh_lmt]
        else:
            # search in y-dirn having almost same x value
            x_vals = list(abs(np.array(unused_centers)[:,0] - pt[0]))
            dist = [[abs((unused_centers[i])[1] - pt[1])+x_vals[i] , i] for i in range(len(x_vals)) if x_vals[i]<neigh_lmt]
            if(len(dist)==0 and len(unused_centers)!=0):
                # If bars are of equal height in group we don't need to search for point in y dirn we can find it in x - dirn
                y_vals = list(abs(np.array(unused_centers)[:,1] - pt[1]))
                dist = [[abs((unused_centers[i])[0] - pt[0])+y_vals[i] , i] for i in range(len(y_vals)) if y_vals[i]<neigh_lmt]
                if(len(dist)==0 and len(unused_centers)!=0) :
                    # bin height is zero
                    bin_height +=[0]
                    bin_width +=[0]
                    bin_centers +=[0]
                    # continuing algorithm from break, as we have zero bins
                    find_next(baseline_pts[base_id+1],False,bin_center,bin_height,bin_width,base_id+1)
                else :
                    #add same as last bin height becoz both have parallel heights
                    bin_height+=[bin_height[len(bin_height)-1]]
                    flag=True
        if(len(dist)>0):
            nxtpt_index = sorted(dist)[0][1]
            nxt_pt = unused_centers[nxtpt_index]
            if(flag):
                # search in x-dirn having almost same y value
                bin_centers += [(nxt_pt[0]+pt[0])//2]
                bin_width += [(nxt_pt[0]-pt[0])]
                flag = False
            else:
                # search in y-dirn having almost same x value
                bin_height += [nxt_pt[1]-base_val]
                flag = True
            return find_next(nxt_pt,flag,bin_centers,bin_height,bin_width,base_id)
    return pt

neigh_lmt=7
#remove all border_pts
i=0
while i<len(unused_centers):
    if( (unused_centers[i][0]<neigh_lmt or unused_centers[i][1]<neigh_lmt) or (unused_centers[i][0]>(X-neigh_lmt) or unused_centers[i][1]>(Y-neigh_lmt)) ):
        unused_centers.remove(unused_centers[i])
        i-=1
    i+=1

#distances of centers from origin to find base point
dist = list(np.array(unused_centers)[:,1]+np.array(unused_centers)[:,0])
first_pt = unused_centers[dist.index(min(dist))]
#distances of centers from origin to find base point
dist = list(np.array(unused_centers)[:,1]+abs(X-np.array(unused_centers)[:,0]))
last_pt = unused_centers[dist.index(min(dist))]

#compute all points belonging to base line and remove it from unused centers
base_val=(first_pt[1]+last_pt[1])//2
heights = list(np.array(unused_centers)[:,1]-base_val)
baseline_pts = [unused_centers[i] for i in range(len(heights)) if(heights[i]<neigh_lmt)]
for i in baseline_pts:
    unused_centers.remove(i)

#TO COMPUTE HEIGHTS AND BIN CENTERS OF HISTOGRAM
bin_center = []
bin_height = []
bin_width =[]
# This base_id last var helps in continuing algorithm from breaks if we have zero bins
end = find_next(baseline_pts[0],False,bin_center,bin_height,bin_width,0)
print(bin_center,bin_height,bin_width)

plt.bar(bin_center,height=bin_height,width=bin_width,bottom=base_val, color=[[0.2,0.2,0.2]])
plt.axis('off')
plt.suptitle("Reconstructed histogram")
plt.savefig('/Users/daggubatisirichandana/PycharmProjects/chart_percept/Topnvispaper/data/histogram/hg32/Reconstructed_hist.png')

plt.show()

