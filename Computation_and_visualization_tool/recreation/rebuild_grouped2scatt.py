import pandas as pd
from  Computation_visualization_tool.Visualizer.colormap import *
import seaborn as sns
from sklearn.cluster import *
import itertools
from operator import add

# data_tensors = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Topnvispaper/Grouped_bar_redesigning/tensor_vote_matrix.csv", sep=",", index_col=False)
# X = len(data_tensors["X"].unique())
# Y = len(data_tensors["Y"].unique())
# print(X,Y)
# cord_list = {
#         "x_val": [],
#         "y_val": [],
#         "CL": []
#     }
#
# a=list(map(add,  data_tensors["val1"], data_tensors["val2"]))
# amin, amax = min(a), max(a)
# for i, val in enumerate(a):
#     a[i] = (val-amin) / (amax-amin)
#
# for i in range(X * Y):
#      if (data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0) and (data_tensors["CL"][i] < 0.5):# and (a[i]>0.02):
#            cord_list["x_val"].append(data_tensors["X"][i])
#            cord_list["y_val"].append(data_tensors["Y"][i])
#            cord_list["CL"].append((data_tensors["CL"][i]))
# # fig, ax = plt.subplots()
# # plt.suptitle("Saliency Visualization")
# # ax.axis('on')
# # Q1 = plt.scatter(cord_list["x_val"], cord_list["y_val"], s=1, c=cord_list['CL'], cmap=colorMap_rbb())
# # plt.colorbar(Q1)
#
# data = np.array([cord_list['x_val'],cord_list['y_val']]).T
#
# db = DBSCAN(eps=7, min_samples=5).fit(data)
# labels = db.labels_
# centers=[]
# for i in (np.unique(labels)[1:]):
#     indexes = [id for id in range(len(labels)) if labels[id] == i]
#     x=0
#     y=0
#     for k in indexes:
#         x+=cord_list['x_val'][k]
#         y+=cord_list['y_val'][k]
#     centers+=[[x//len(indexes),y//len(indexes)]]
# centers=sorted(centers)
# unused_centers=sorted(centers)
#
# # df1= pd.DataFrame(cord_list['x_val'],columns=["x"])
# # df2= pd.DataFrame(cord_list['y_val'], columns=["y"])
# # df3= pd.DataFrame(db.labels_ ,columns=["label"])
# # df = pd.concat([df1,df2,df3], axis=1)
# # #plot data with seaborn
# # facet = sns.lmplot(data=df, x='x', y='y', hue='label',
# #                    fit_reg=False, legend=True, legend_out=True)
# # plt.suptitle("Clustering points with black dots as cluster centers")
# # plt.plot(np.array(centers)[:,0], np.array(centers)[:,1], 'ko')
#
# print("CENTERS:\n", centers)

# grouped bar
centers = [[281, 151], [283, 406], [386, 407], [388, 152], [389, 662], [493, 152], [493, 663], [599, 152], [600, 576], [702, 577], [705, 152], [706, 618], [809, 618], [810, 152], [916, 152], [917, 530], [1019, 531], [1021, 152], [1022, 577], [1126, 576], [1127, 152], [1233, 152], [1234, 576], [1338, 152], [1338, 576], [1340, 531], [1443, 531], [1444, 152], [1550, 152], [1550, 663], [1655, 152], [1655, 661], [1760, 662], [1761, 151]]
unused_centers =  [[281, 151], [283, 406], [386, 407], [388, 152], [389, 662], [493, 152], [493, 663], [599, 152], [600, 576], [702, 577], [705, 152], [706, 618], [809, 618], [810, 152], [916, 152], [917, 530], [1019, 531], [1021, 152], [1022, 577], [1126, 576], [1127, 152], [1233, 152], [1234, 576], [1338, 152], [1338, 576], [1340, 531], [1443, 531], [1444, 152], [1550, 152], [1550, 663], [1655, 152], [1655, 661], [1760, 662], [1761, 151]]
X=1912
Y=962
def find_next(pt,flag,bin_centers,bin_height):
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
        if(len(dist)>0):
            nxtpt_index = sorted(dist)[0][1]
            nxt_pt = unused_centers[nxtpt_index]
            if(flag):
                # search in x-dirn having almost same y value
                bin_centers += [(nxt_pt[0]+pt[0])//2]
                flag = False
            else:
                # search in y-dirn having almost same x value
                bin_height += [nxt_pt[1]-base_y]
                flag = True
            return find_next(nxt_pt,flag,bin_centers,bin_height)
    return pt



base_y=centers[0][1]
groups_centers = []
groups_height = []
base_pts = [c for c in centers if (c[1]-base_y) < 5]


neigh_lmt=5
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
bin_width=(baseline_pts[1][0]-baseline_pts[0][0])

end=[0,0]
groups_height=[]
groups_centers=[]
while(len(unused_centers)>0 and end!=last_pt):
    bin_center = []
    bin_height = []
    for i,val in enumerate(np.array(baseline_pts)[:,0]-end[0]):
        if val>neigh_lmt:
            id=i
            break
    end = find_next(baseline_pts[id],False,bin_center,bin_height)
    # If bars are of equal height in group we don't need to search for point in y dirn we can find it in x - dirn
    if(len(groups_height)>0 and len(bin_height) < np.max([len(a) for a in groups_height])):
        bin_height+=[bin_height[len(bin_height)-1]]
        end = find_next(end,True,bin_center,bin_height)
        if(len(bin_height)!=len(bin_center)):
            bin_height+=[0]
            for i,val in enumerate(np.array(baseline_pts)[:,0]-end[0]):
                if val>neigh_lmt:
                    id=i
                    break
            bin_center+=[(end[0]+baseline_pts[id][0])//2]
            end = find_next(baseline_pts[id],False,bin_center,bin_height)
    print(bin_center,bin_height)
    groups_centers += [bin_center]
    groups_height += [bin_height]

groups_height=np.array(groups_height)
groups_centers=np.array(groups_centers)
# To make other heights as zeros
max_len = np.max([len(a) for a in groups_height])
groups_height=np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in groups_height])
max_len = np.max([len(a) for a in groups_centers])
groups_centers=np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in groups_centers])

# # plot grouped bar
# for i in range(len(groups_height[0])):
#     plt.bar(groups_centers[:,i],height=groups_height[:,i],width=bin_width,bottom=base_y)
# plt.axis('off')
# plt.suptitle("Reconstructed grouped bar")
# plt.savefig('/Users/daggubatisirichandana/PycharmProjects/chart_percept/Topnvispaper/Grouped_bar2/Reconstructed_groupedbar.png')

title=['ID1','ID2','ID3','ID4','ID5']

# plot scatter plot of groups
plt.plot(groups_height[:,0], groups_height[:,1], 'ko')
for i, txt in enumerate(title):
    # plt.annotate(txt, (groups_height[:,0][i], groups_height[:,1][i]))
    plt.text(groups_height[:,0][i]+1.5, groups_height[:,1][i]+1.5, txt, fontsize=6)
plt.xlabel("Category 1 (in pixels)")
plt.ylabel("Category 2 (in pixels)")
plt.grid()
plt.suptitle("Redesigned scatter plot of Category 1 and Category 2 ")
plt.savefig('/Users/daggubatisirichandana/PycharmProjects/chart_percept/Topnvispaper/Grouped_bar_redesigning/redesigned_scatter.png')


# # plot groups seperately
# plt.grid(zorder=0)
# plt.bar(groups_centers[:,0],height=groups_height[:,0],width=bin_width,bottom=base_y, color=[[0.2,0.2,0.2]], zorder=3)
# plt.xticks(groups_centers[:,0], title)
# plt.ylabel("(in pixels)")
# plt.suptitle("Redesigned Bar Chart of Category 1")
# plt.savefig('/Users/daggubatisirichandana/PycharmProjects/chart_percept/Topnvispaper/Grouped_bar_redesigning/Redesigned_bar1.png')

# plt.grid(zorder=0)
# plt.bar(groups_centers[:,1],height=groups_height[:,1],width=bin_width,bottom=base_y, color=[[0.2,0.2,0.2]], zorder=3)
# plt.xticks(groups_centers[:,1], title)
# plt.ylabel("(in pixels)")
# plt.suptitle("Redesigned Bar Chart of Category 2")
# plt.savefig('/Users/daggubatisirichandana/PycharmProjects/chart_percept/Topnvispaper/Grouped_bar_redesigning/Redesigned_bar2.png')


plt.show()
