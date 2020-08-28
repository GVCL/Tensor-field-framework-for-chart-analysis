import xml.etree.ElementTree as ET
from Computation_and_visualization_tool.Text_DET_REC.Retrieve_Text import get_text_labels,get_title,get_xtitle,get_ytitle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import *
from operator import add
import seaborn as sns
import csv
import cv2
import os

def bar(filename):
    image_name = os.path.basename(filename).split(".png")[0]
    path = os.path.dirname(filename)+'/'
    data_tensors = pd.read_csv(path+"tensor_vote_matrix_"+image_name+".csv", sep=",", index_col=False)
    X = len(data_tensors["X"].unique())
    Y = len(data_tensors["Y"].unique())
    # print(X,Y)
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
         if (data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0) and (data_tensors["CL"][i] < 0.5) and (a[i]>0.002):
             if(data_tensors["X"][i]<(X-5) and data_tensors["X"][i]>5 and data_tensors["Y"][i]>5 and data_tensors["Y"][i]<(Y-5)) :
                 cord_list["x_val"].append(data_tensors["X"][i])
                 cord_list["y_val"].append(data_tensors["Y"][i])
                 cord_list["CL"].append((data_tensors["CL"][i]))

    data = np.array([cord_list['x_val'],cord_list['y_val']]).T

    db = DBSCAN(eps=5, min_samples=2).fit(data)
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

    # # plot data with seaborn
    # plt.scatter(cord_list["x_val"],  cord_list["y_val"], s=10, c='b')
    # plt.suptitle("Degenerate points and cluster centers")
    # plt.scatter(np.array(centers)[:,0], np.array(centers)[:,1], s=10, c='r', alpha=0.7)
    # plt.axis('off')

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
    bar_width=[]
    while i<len(unused_centers):
        if (i!=len(unused_centers)-1) and (abs(unused_centers[i][1]-unused_centers[i+1][1])<neigh_lmt):
            bin_heights+=[unused_centers[i][1]-base_val]
            bin_centers+=[(unused_centers[i][0]+unused_centers[i+1][0])//2]
            bar_width+=[abs(unused_centers[i][0]-unused_centers[i+1][0])//2]
            i+=2
        else:
            bin_heights+=[unused_centers[i][1]-base_val]
            bin_centers+=[unused_centers[i][0]]
            i+=1

    # In this case we don't have two corner points for as they are very thin
    bar_width = 2*np.mean(bar_width)
    if(len(unused_centers)<len(bin_heights)*2):
        i=0
        bar_width = 10
        bin_heights=[]
        bin_centers=[]
        while i<len(unused_centers):
            bin_heights+=[unused_centers[i][1]-base_val]
            bin_centers+=[unused_centers[i][0]]
            i+=1
    # TO remove zero heights
    zero_ids = list(np.where(np.array(bin_heights) == 0)[0])
    bin_height = list(np.delete(np.array(bin_heights), zero_ids))
    bin_center = list(np.delete(np.array(bin_centers), zero_ids))
    bar_width = list(np.delete(np.array(bar_width), zero_ids))
    # print(bin_heights,bin_centers)

    # To get label text
    img = cv2.imread(path+image_name+".png")
    root = ET.parse(path+image_name+'.xml').getroot()
    Xlabel,Ylabel,xbox_centers,ybox_centers = get_text_labels(img,root)
    if(isinstance(Ylabel[0], str) and Ylabel[0].isnumeric()):
        Ylabel = [int(i) for i in Ylabel if i.isnumeric()]
    # To deal with duplicate values and make one as negative
    for i in np.unique(Ylabel):
        id=[j for j, val in enumerate(Ylabel) if i==val]
        if(len(id)==2):
            if(ybox_centers[id[0]][1]>ybox_centers[id[1]][1]):
                Ylabel[id[1]]*=-1
                neg_ids=np.where(ybox_centers[:,1] < ybox_centers[id[1]][1])[0]
            else:
                Ylabel[id[0]]*=-1
                neg_ids=np.where(ybox_centers[:,0] < ybox_centers[id[0]][1])[0]
            for i in neg_ids:
                Ylabel[i]*=-1
    normalize_scaley =abs((Ylabel[0]-Ylabel[1])/(ybox_centers[0][1]-ybox_centers[1][1]))
    bin_heights = np.array(bin_heights)
    first_label = Ylabel[(np.where(ybox_centers[:,1] == max(ybox_centers[:,1])))[0][0]]
    first_label -= (Y-max(ybox_centers[:,1])-base_val)*normalize_scaley
    bin_heights = np.array(bin_heights)*normalize_scaley + round(first_label)
    if str(sorted(Xlabel)[0]).isnumeric():
        # To deal with duplicate values and make one as negative
        for i in np.unique(Xlabel):
            id=[j for j, val in enumerate(Xlabel) if i==val]
            if(len(id)==2):
                if(xbox_centers[id[0]][1]>xbox_centers[id[1]][1]):
                    Xlabel[id[1]]*=-1
                    neg_ids=np.where(xbox_centers[:,1] < xbox_centers[id[1]][1])[0]
                else:
                    Xlabel[id[0]]*=-1
                    neg_ids=np.where(xbox_centers[:,0] < xbox_centers[id[0]][1])[0]
                for i in neg_ids:
                    Xlabel[i]*=-1

        normalize_scalex = (Xlabel[0]-Xlabel[1])/(xbox_centers[0][0]-xbox_centers[1][0])
        first_label=Xlabel[(np.where(xbox_centers[:,0]==min(xbox_centers[:,0])))[0][0]]
        bin_centers = np.array(bin_centers) - min(xbox_centers[:,0])
        bin_centers = np.array(bin_centers)*normalize_scalex + first_label
        bar_width = np.array(bar_width)*normalize_scalex
        labels = bin_centers
    else:
        i=0
        labels=[]
        while i<len(bin_centers):
            ids = [j for j,val in enumerate(np.array(xbox_centers)[:,0]) if (val>bin_centers[i]-bar_width[0]//2 and val<bin_centers[i]+bar_width[0]//2)]
            if(len(ids)!=0):
                labels+=[' '.join([Xlabel[j] for j in ids])]
                xbox_centers = np.delete(xbox_centers,ids,axis=0)
                Xlabel = np.delete(Xlabel,ids)
            else:
                # IF for a given height there is no x label detected we add a default label
                labels+=['NO LABEL']
            i+=1
        # IF there are still some labels left then the bar height is zero
        i=0
        bin_heights=bin_heights.tolist()
        while i<len(xbox_centers):
            bin_centers+=[int(xbox_centers[i][0])]
            labels+=[Xlabel[i]]
            bin_heights+=[0]
            i+=1
        plt.xticks(bin_centers, labels, rotation=90, fontsize=10)
    # print(bin_centers,bin_heights)

    # Reconstruct bar
    plt.bar(bin_centers,height=bin_heights,width=bar_width, color=[[0.2,0.2,0.2]])
    plt.xlabel(get_xtitle(img,root))
    plt.ylabel(get_ytitle(img,root))
    plt.title(get_title(img,root))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path+'reconstructed_'+image_name+".png")

    # Writing data to CSV file
    L = [['X','Y','title','x-title','y-title']]
    L = L + [[labels[0], bin_heights[0], get_title(img,root), get_xtitle(img,root), get_ytitle(img,root)]]
    L = L + [[labels[i], bin_heights[i]] for i in range(1,len(labels))]
    with open(path+'data_'+image_name+'.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(L)


