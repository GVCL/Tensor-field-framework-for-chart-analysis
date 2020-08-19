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

def reconstruct_bar():
    data_tensors = pd.read_csv("/home/komaldadhich/Desktop/test_results/bars/bars/TB_img/tensor_vote_matrix.csv", sep=",", index_col=False)
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
         if (data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0) and (data_tensors["CL"][i] < 0.4) and (a[i]>0.005):
             if(data_tensors["X"][i]<(X-5) and data_tensors["X"][i]>5 and data_tensors["Y"][i]>5 and data_tensors["Y"][i]<(Y-5)) :
                 cord_list["x_val"].append(data_tensors["X"][i])
                 cord_list["y_val"].append(data_tensors["Y"][i])
                 cord_list["CL"].append((data_tensors["CL"][i]))

    fig, ax = plt.subplots()
    plt.suptitle("Saliency Visualization")
    ax.axis('on')
    plt.scatter(cord_list["x_val"], cord_list["y_val"], s=1)
    # plt.colorbar(Q1)
    plt.show()

    data = np.array([cord_list['x_val'],cord_list['y_val']]).T

    # db = DBSCAN(eps=5, min_samples=3).fit(data)
    db = DBSCAN(eps=7, min_samples=3).fit(data)
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
    # plt.plot(np.array(centers)[:,0], np.array(centers)[:,1], 'ko')

    # Reconstruction of bar
    unused_centers=sorted(centers)
    #distances of centers from origin to find base point
    dist = list(np.array(unused_centers)[:,1]+np.array(unused_centers)[:,0])
    first_pt = unused_centers[dist.index(min(dist))]
    #distances of centers from origin to find base point
    dist = list(np.array(unused_centers)[:,1]+abs(X-np.array(unused_centers)[:,0]))
    last_pt = unused_centers[dist.index(min(dist))]
    base_val=(first_pt[1]+last_pt[1])//2

    neigh_lmt=5
    heights = list(abs(np.array(unused_centers)[:,1]-base_val))
    unused_centers = [unused_centers[i] for i in range(len(heights)) if(heights[i]>neigh_lmt)]

    i=0
    bin_heights=[]
    bin_centers=[]
    while i<len(unused_centers):
        if (i!=len(unused_centers)-1) and (abs(unused_centers[i][1]-unused_centers[i+1][1])<neigh_lmt):
            bin_heights+=[unused_centers[i][1]-base_val]
            bin_centers+=[(unused_centers[i][0]+unused_centers[i+1][0])//2]
            i+=2
        else:
            bin_heights+=[unused_centers[i][1]-base_val]
            bin_centers+=[unused_centers[i][0]]
            i+=1

    # In this case we don't have two corner points for as they are very thin
    bar_witdh = 10
    if(len(unused_centers)<len(bin_heights)*2):
        i=0
        bin_heights=[]
        bin_centers=[]
        while i<len(unused_centers):
            bin_heights+=[unused_centers[i][1]-base_val]
            bin_centers+=[unused_centers[i][0]]
            i+=1
    else:
        bar_witdh = (min(bin_centers)-first_pt[0])*2


    print(bin_heights,bin_centers)


    plt.bar(bin_centers,height=bin_heights,width=bar_witdh,bottom=base_val, color=[[0.2,0.2,0.2]])
    plt.axis('off')
    plt.suptitle("Reconstructed bar")
    plt.savefig('bc021_re.png')

    plt.show()

def main():
    reconstruct_bar()

if __name__ == '__main__':
    main()
