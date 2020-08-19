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


def hist(filename):
    image_name = os.path.basename(filename).split(".png")[0]
    path = os.path.dirname(filename)+'/'
    data_tensors = pd.read_csv(path+"tensor_vote_matrix.csv", sep=",", index_col=False)

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
         if (data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0) and (data_tensors["CL"][i] < 0.4) and (a[i]>0.002):
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
    centers = sorted(centers)
    unused_centers = sorted(centers)
    # # plot data with seaborn
    # plt.scatter(cord_list["x_val"],  cord_list["y_val"], s=10, c='b')
    # plt.suptitle("Degenerate points and cluster centers")
    # plt.scatter(np.array(centers)[:,0], np.array(centers)[:,1], s=10, c='r', alpha=0.7)
    # plt.axis('off')

    """ Histogram """
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
                    bin_width += [abs(nxt_pt[0]-pt[0])]
                    flag = False
                else:
                    # search in y-dirn having almost same x value
                    bin_height += [nxt_pt[1]-base_val]
                    flag = True
                return find_next(nxt_pt,flag,bin_centers,bin_height,bin_width,base_id)
        return pt


    neigh_lmt=7
    #remove all corner pts in img
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
    # TO remove zero heights
    zero_ids = list(np.where(np.array(bin_height) == 0)[0])
    bin_height = list(np.delete(np.array(bin_height), zero_ids))
    bin_center = list(np.delete(np.array(bin_center), zero_ids))
    bin_width = list(np.delete(np.array(bin_width), zero_ids))
    bin_width = np.array(bin_width)+4
    if((len(bin_height)-len(bin_center))==1):
        bin_height=bin_height[:len(bin_height)-1]
    # print(bin_center,bin_height)

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
    bin_height = np.array(bin_height)
    first_label = Ylabel[(np.where(ybox_centers[:,1] == max(ybox_centers[:,1])))[0][0]]
    first_label -= (Y-max(ybox_centers[:,1])-base_val)*normalize_scaley
    bin_height = np.array(bin_height)*normalize_scaley + round(first_label)

    if(isinstance(Xlabel[0], str) and Xlabel[0].isnumeric()):
        Xlabel = [int(i) for i in Xlabel if i.isnumeric()]
    # To deal with duplicate values and make one as negative
    for i in np.unique(Xlabel):
        id=[j for j, val in enumerate(Xlabel) if i==val]
        if(len(id)==2):
            if(xbox_centers[id[0]][0]>xbox_centers[id[1]][0]):
                Xlabel[id[1]]*=-1
                neg_ids=np.where(xbox_centers[:,0] < xbox_centers[id[1]][0])[0]
            else:
                Xlabel[id[0]]*=-1
                neg_ids=np.where(xbox_centers[:,0] < xbox_centers[id[0]][0])[0]
            for i in neg_ids:
                Xlabel[i]*=-1
    normalize_scalex = (Xlabel[0]-Xlabel[1])/(xbox_centers[0][0]-xbox_centers[1][0])
    bin_center = np.array(bin_center) - min(xbox_centers[:,0])
    first_label=Xlabel[(np.where(xbox_centers[:,0]==min(xbox_centers[:,0])))[0][0]]
    bin_center = np.array(bin_center)*normalize_scalex + first_label
    bin_width = np.array(bin_width)*normalize_scalex

    # Reconstruct bar
    # plt.figure(figsize=(X//100,Y//100+3))
    plt.bar(bin_center,height=bin_height,width=bin_width, color=[[0.2,0.2,0.2]])
    plt.xlabel(get_xtitle(img,root))
    plt.ylabel(get_ytitle(img,root))
    plt.title(get_title(img,root))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path+'reconstructed_'+image_name+".png")

    # Writing data to CSV file
    L = [['bin_center','freq','bin_width','title','x-title','y-title']]
    L = L + [[bin_center[0], bin_height[0], bin_width[0], get_title(img,root), get_xtitle(img,root), get_ytitle(img,root)]]
    L = L + [[bin_center[i], bin_height[i], bin_width[i]] for i in range(1,len(bin_center))]
    with open(path+'data.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(L)


    # plt.show()
