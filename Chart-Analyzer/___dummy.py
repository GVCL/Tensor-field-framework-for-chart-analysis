import xml.etree.ElementTree as ET
from Retrieve_Text import get_text_labels,get_title,get_xtitle,get_ytitle,get_legends
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

def G_bar(filename):
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
         if (data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0) and (data_tensors["CL"][i] < 0.5) and (a[i]>0.001):
             if(data_tensors["X"][i]<(X-5) and data_tensors["X"][i]>5 and data_tensors["Y"][i]>5 and data_tensors["Y"][i]<(Y-5)) :
                 cord_list["x_val"].append(data_tensors["X"][i])
                 cord_list["y_val"].append(data_tensors["Y"][i])
                 cord_list["CL"].append((data_tensors["CL"][i]))

    data = np.array([cord_list['x_val'],cord_list['y_val']]).T
    db = DBSCAN(eps=4, min_samples=3).fit(data)
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
    unused_centers = sorted(centers)
    # # plot data with seaborn
    # plt.scatter(cord_list["x_val"],  cord_list["y_val"], s=10, c='b')
    # plt.suptitle("Degenerate points and cluster centers")
    # plt.scatter(np.array(centers)[:,0], np.array(centers)[:,1], s=10, c='r', alpha=0.7)
    # plt.axis('off')

    # Reconstruction of Grouped Bar
    ''' Compute all points belonging to base line and remove it from unused centers '''
    neigh_lmt=5
    #distances of centers from origin to find base point
    dist = list(np.array(unused_centers)[:,1]+np.array(unused_centers)[:,0])
    first_pt = unused_centers[dist.index(min(dist))]
    #distances of centers from origin to find base point
    dist = list(np.array(unused_centers)[:,1]+abs(X-np.array(unused_centers)[:,0]))
    last_pt = unused_centers[dist.index(min(dist))]
    base_val=(first_pt[1]+last_pt[1])//2
    heights = list(np.array(unused_centers)[:,1]-base_val)
    baseline_pts = [unused_centers[i] for i in range(len(heights)) if(heights[i]<=neigh_lmt)]
    for i in baseline_pts:
        unused_centers.remove(i)

    ''' FIND ALL BAR HEIGHTS AND BAR CENTERS'''
    i=0
    bar_heights=[]
    bar_centers=[]
    while i<len(unused_centers):
        flag=False
        for j in range(i+1,len(unused_centers)):

            if(len(bar_heights)==0):
                flag = True
                if(abs(unused_centers[j][1]-unused_centers[i][1])<8):
                    if abs(unused_centers[j][0]-unused_centers[i][0])>10:
                        #almost equal height
                        bar_heights+=[unused_centers[i][1]-base_val]
                        bar_centers+=[(unused_centers[i][0]+unused_centers[j][0])//2]
                        bar_width = abs(unused_centers[i][0]-unused_centers[j][0])
                        unused_centers = np.delete(unused_centers,[i,j],axis=0)
                        i-=1
                    break
            elif(abs(unused_centers[j][0]-unused_centers[i][0]-bar_width)<10 and abs(unused_centers[j][1]-unused_centers[i][1])<7):
                bar_heights+=[unused_centers[i][1]-base_val]
                bar_centers+=[(unused_centers[i][0]+unused_centers[j][0])//2]
                bar_width=(bar_width+(abs(unused_centers[i][0]-unused_centers[j][0])))/2
                unused_centers = np.delete(unused_centers,[i,j],axis=0)
                flag = True
                i-=1
                break
        if flag == False:
            # If we have equal height as prev bar
            if(abs(bar_heights[len(bar_heights)-1]+base_val-unused_centers[i][1])<=12 and abs(abs(unused_centers[i][0]-bar_centers[len(bar_centers)-1])-(3*bar_width/2))<10):
                bar_heights+=[unused_centers[i][1]-base_val]
                bar_centers+=[unused_centers[i][0]-(bar_width//2)]
                unused_centers = np.delete(unused_centers,[i],axis=0)
                i-=1
        i+=1
    bar_width = bar_width+3 # adding pix width
    # print(bar_heights,bar_centers)

    ''' To get legend colors and its labels'''
    img = cv2.imread(path+image_name+".png")
    root = ET.parse(path+image_name+'.xml').getroot()
    # Now group stack heights based on it's catogery
    group_colors,group_leg_labels = get_legends(img,root)
    group_heights = []
    group_center = []
    group_id=[]
    remove_ids=[]
    for i in range(len(bar_centers)):
        if (img[int(Y-(bar_heights[i]//2+base_val)),int(bar_centers[i])]).tolist() in group_colors:
            group_id += [group_colors.index((img[int(Y-(bar_heights[i]//2+base_val)),int(bar_centers[i])]).tolist())]
        else:
            remove_ids+=[i]
    bar_centers = np.delete(np.array(bar_centers),remove_ids,axis=0)
    bar_heights = np.delete(np.array(bar_heights),remove_ids)
    i=0
    remove_ids=[]
    group_id_order = list(np.unique(group_id))
    while i<len(group_id):
        ln_gc = len(group_colors)
        h = [0]*len(group_colors)
        if i+ln_gc<=len(group_id) and sorted(group_id[i:i+ln_gc]) == list(range(ln_gc)):
            # Check  if they equidistant, If not they must be from other batch
            # 5 pix is assumed as thin gap between bars in a batch
            center_diff = [abs(bar_centers[j]-bar_centers[j+1])-bar_width for j in range(i,i+ln_gc-1)]
            if(max(center_diff)<8):
                group_id_order=group_id[i:i+ln_gc]
                for j in range(i,i+ln_gc):
                    h[group_id[j]] = bar_heights[j]
                group_heights+=[h]
                group_center+=[np.mean(bar_centers[i:i+ln_gc]) ]
                remove_ids+=range(i,i+ln_gc)
                i+=ln_gc
            else:
                i+=1
        else:
            i+=1
    bar_centers = np.delete(np.array(bar_centers),remove_ids,axis=0)
    bar_heights = np.delete(np.array(bar_heights),remove_ids)
    group_id = np.delete(np.array(group_id),remove_ids)
    # print(group_heights,group_center,group_id)
    i=0
    while i<len(group_id):
        # print(bar_centers[i])
        bar_ctr=bar_centers[i]-(((1-ln_gc)/2)+group_id_order.index(group_id[i]))*bar_width
        temp = [i for i,val in enumerate(bar_centers) if val>(bar_ctr-(ln_gc/2)*bar_width) and val<(bar_ctr+(ln_gc/2)*bar_width)]
        h = [0]*len(group_colors)
        for j in temp:
            h[group_id[j]] = bar_heights[j]
        group_heights+=[h]
        group_center+=[bar_ctr]
        bar_centers = np.delete(np.array(bar_centers),temp,axis=0)
        bar_heights = np.delete(np.array(bar_heights),temp)
        group_id = np.delete(np.array(group_id),temp)
    # print(group_colors,group_leg_labels,group_heights,group_center)

    ''' Map pixels to original coordinates'''
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
    group_heights = np.array(group_heights)
    first_label = Ylabel[(np.where(ybox_centers[:,1] == max(ybox_centers[:,1])))[0][0]]
    first_label -= (Y-max(ybox_centers[:,1])-base_val)*normalize_scaley
    # group_heights = np.array(group_heights)*normalize_scaley + round(first_label)
    group_heights = np.array(group_heights).astype(np.float32)
    for i in range(len(group_heights)):
        for j in range(len(h)):
            if group_heights[i][j] != 0:
                group_heights[i][j] = (group_heights[i][j]*normalize_scaley) + first_label
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
        group_center = np.array(group_center) - min(xbox_centers[:,0])
        group_center = np.array(group_center)*normalize_scalex + first_label
        bar_width = np.array(bar_width)*normalize_scalex
        labels = group_center
    else:
        i=0
        labels=[]
        while i<len(group_center):
            ids = [j for j,val in enumerate(np.array(xbox_centers)[:,0]) if (val>group_center[i]-(ln_gc/2)*bar_width and val<group_center[i]+(ln_gc/2)*bar_width)]
            if(len(ids)!=0):
                labels+=[' '.join([Xlabel[j] for j in ids])]
                xbox_centers = np.delete(xbox_centers,ids,axis=0)
                Xlabel = np.delete(Xlabel,ids)
            else:
                # IF for a given height there is no x label detected we add a default label
                labels+=[' No label ']
            i+=1
        # IF there are still some labels left then the bar height is zero
        i=0
        group_heights = group_heights.tolist()
        while i<len(xbox_centers):
            group_center+=[int(xbox_centers[i][0])]
            labels+=[Xlabel[i]]
            group_heights+=[[0]*len(group_heights[0])]
            i+=1
        plt.xticks(group_center, labels, rotation=90, fontsize=10)
    # print(group_colors,group_leg_labels,group_heights,group_center)


    '''Reconstruct Grouped bar'''
    group_heights = np.array(group_heights)
    for i in range(len(group_colors)):
            k = ((1-len(group_colors))/2+i)*bar_width
            plt.bar(np.array(group_center)+k,height=group_heights[:,i], width=bar_width, color=[np.array(group_colors[i][::-1])/255],edgecolor='k', label=group_leg_labels[i])
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.xlabel(get_xtitle(img,root))
    plt.ylabel(get_ytitle(img,root))
    plt.title(get_title(img,root))
    plt.tight_layout()
    plt.savefig(path+'reconstructed_'+image_name+".png")


    # Writing data to CSV file
    L = [['X']+group_leg_labels+['bar_width','title','x-title','y-title']]
    L = L + [[labels[0]]+group_heights[0].tolist()+[bar_width, get_title(img,root), get_xtitle(img,root), get_ytitle(img,root)]]
    L = L + [[labels[i]]+group_heights[i].tolist() for i in range(1,len(labels))]
    with open(path+'data_'+image_name+'.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(L)

def H_bar(filename):
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
         if (data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0) and (data_tensors["CL"][i] < 0.5) and (a[i]>0.005):
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

    #swap x and y co-ordinates and compute height same as for horizontal bars
    centers=np.array(centers)
    centers[:,[0, 1]] = centers[:,[1, 0]]
    centers = centers.tolist()

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
    # plt.barh(bin_centers, width=bin_heights, height=bar_width, color=[[0.2,0.2,0.2]])
    # plt.axis('off')
    # plt.suptitle("Reconstructed pixel bar")


    # To get label text
    img = cv2.imread(path+image_name+".png")
    root = ET.parse(path+image_name+'.xml').getroot()
    Xlabel,Ylabel,xbox_centers,ybox_centers = get_text_labels(img,root)
    ybox_centers= np.array([[ybox_centers[i][0],abs(ybox_centers[i][1]-Y)] for i in range(len(ybox_centers))])
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
    first_label=Xlabel[(np.where(xbox_centers[:,0]==min(xbox_centers[:,0])))[0][0]]
    first_label -= (min(xbox_centers[:,0])-base_val)*normalize_scalex
    bin_heights = np.array(bin_heights)*normalize_scalex + round(first_label)

    if str(sorted(Ylabel)[0]).isnumeric():
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
        first_label = Ylabel[(np.where(ybox_centers[:,1] == min(ybox_centers[:,1])))[0][0]]
        bin_centers = np.array(bin_centers) - min(ybox_centers[:,1])
        bin_centers = np.array(bin_centers)*normalize_scaley + round(first_label)
        bar_width = np.array(bar_width)*normalize_scaley
        labels = bin_centers
    else:
        i=0
        labels=[]
        while i<len(bin_centers):
            ids = [j for j,val in enumerate(np.array(ybox_centers)[:,1]) if (val>bin_centers[i]-bar_width[0]//2 and val<bin_centers[i]+bar_width[0]//2)]
            if(len(ids)!=0):
                labels+=[' '.join([str(Ylabel[j]) for j in ids])]
                ybox_centers = np.delete(ybox_centers,ids,axis=0)
                Ylabel = np.delete(Ylabel,ids)
            else:
                # IF for a given height there is no x label detected we add a default label
                labels+=['NO LABEL']
            i+=1
        # IF there are still some labels left then the bar height is zero
        i=0
        bin_heights=bin_heights.tolist()
        while i<len(ybox_centers):
            bin_centers+=[int(ybox_centers[i][1])]
            labels+=[Ylabel[i]]
            bin_heights+=[0]
            i+=1
        plt.yticks(bin_centers, labels, fontsize=10)

    # Reconstruct bar
    plt.barh(bin_centers,width=bin_heights,height=bar_width, color=[[0.2,0.2,0.2]])
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

def GH_bar(filename):
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
    db = DBSCAN(eps=4, min_samples=3).fit(data)
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


    #swap x and y co-ordinates and compute height same as for horizontal bars
    centers=np.array(centers)
    centers[:,[0, 1]] = centers[:,[1, 0]]
    centers = centers.tolist()
    unused_centers = sorted(centers)
    #
    # # plot data with seaborn
    # plt.scatter(cord_list["y_val"],  cord_list["x_val"], s=10, c='b')
    # plt.suptitle("Degenerate points and cluster centers")
    # plt.scatter(np.array(centers)[:,0], np.array(centers)[:,1], s=10, c='r', alpha=0.7)
    # plt.axis('off')

    # horizontal grouped bar
    '''compute all points belonging to base line and remove it from unused centers'''
    neigh_lmt=5
    #distances of centers from origin to find base point
    dist = list(np.array(unused_centers)[:,1]+np.array(unused_centers)[:,0])
    first_pt = unused_centers[dist.index(min(dist))]
    #distances of centers from origin to find base point
    dist = list(np.array(unused_centers)[:,1]+abs(X-np.array(unused_centers)[:,0]))
    last_pt = unused_centers[dist.index(min(dist))]
    base_val=(first_pt[1]+last_pt[1])//2
    heights = list(np.array(unused_centers)[:,1]-base_val)
    baseline_pts = [unused_centers[i] for i in range(len(heights)) if(heights[i]<=neigh_lmt)]
    for i in baseline_pts:
        unused_centers.remove(i)

    ''' FIND ALL BAR HEIGHTS AND BAR CENTERS'''
    i=0
    bar_heights=[]
    bar_centers=[]
    while i<len(unused_centers):
        flag=False
        for j in range(i+1,len(unused_centers)):
            if(len(bar_heights)==0):
                flag = True
                if(abs(unused_centers[j][1]-unused_centers[i][1])<8):
                    if abs(unused_centers[j][0]-unused_centers[i][0])>10:
                        #almost equal height
                        bar_heights+=[unused_centers[i][1]-base_val]
                        bar_centers+=[(unused_centers[i][0]+unused_centers[j][0])//2]
                        bar_width = abs(unused_centers[i][0]-unused_centers[j][0])
                        unused_centers = np.delete(unused_centers,[i,j],axis=0)
                        i-=1
                    break
            elif(abs(unused_centers[j][0]-unused_centers[i][0]-bar_width)<8 and abs(unused_centers[j][1]-unused_centers[i][1])<8):
                bar_heights+=[unused_centers[i][1]-base_val]
                bar_centers+=[(unused_centers[i][0]+unused_centers[j][0])//2]
                bar_width=(bar_width+(abs(unused_centers[i][0]-unused_centers[j][0])))/2
                unused_centers = np.delete(unused_centers,[i,j],axis=0)
                flag = True
                i-=1
                break
        if flag == False:
            # If we have equal height as prev bar
            if(abs(bar_heights[len(bar_heights)-1]+base_val-unused_centers[i][1])<=12 and abs(abs(unused_centers[i][0]-bar_centers[len(bar_centers)-1])-(3*bar_width/2))<10):
                bar_heights+=[unused_centers[i][1]-base_val]
                bar_centers+=[unused_centers[i][0]-(bar_width//2)]
                unused_centers = np.delete(unused_centers,[i],axis=0)
                i-=1
        i+=1
    bar_width = bar_width+3 # adding pix width
    # print(bar_heights,bar_centers)

    ''' To get legend colors and its labels'''
    img = cv2.imread(path+image_name+".png")
    root = ET.parse(path+image_name+'.xml').getroot()
    # Now group stack heights based on it's catogery
    group_colors,group_leg_labels = get_legends(img,root)
    group_heights = []
    group_center = []
    group_id=[]
    remove_ids=[]
    for i in range(len(bar_centers)):
        if (img[int(Y-bar_centers[i]),bar_heights[i]//2+int(base_val)]).tolist() in group_colors:
            group_id += [group_colors.index(img[int(Y-bar_centers[i]),bar_heights[i]//2+int(base_val)].tolist())]
        else:
            remove_ids+=[i]
    bar_centers = np.delete(np.array(bar_centers),remove_ids,axis=0)
    bar_heights = np.delete(np.array(bar_heights),remove_ids)
    i=0
    remove_ids=[]
    group_id_order = list(np.unique(group_id))
    while i<len(group_id):
        ln_gc = len(group_colors)
        h = [0]*len(group_colors)
        if i+ln_gc<=len(group_id) and sorted(group_id[i:i+ln_gc]) == list(range(ln_gc)):
            # Check  if they equidistant, If not they must be from other batch
            # 5 pix is assumed as thin gap between bars in a batch
            center_diff = [abs(bar_centers[j]-bar_centers[j+1])-bar_width for j in range(i,i+ln_gc-1)]
            if(max(center_diff)<8):
                group_id_order=group_id[i:i+ln_gc]
                for j in range(i,i+ln_gc):
                    h[group_id[j]] = bar_heights[j]
                group_heights+=[h]
                group_center+=[np.mean(bar_centers[i:i+ln_gc]) ]
                remove_ids+=range(i,i+ln_gc)
                i+=ln_gc
            else:
                i+=1
        else:
            i+=1
    bar_centers = np.delete(np.array(bar_centers),remove_ids,axis=0)
    bar_heights = np.delete(np.array(bar_heights),remove_ids)
    group_id = np.delete(np.array(group_id),remove_ids)
    i=0
    while i<len(group_id):
        bar_ctr=bar_centers[i]-(((1-ln_gc)/2)+group_id_order.index(group_id[i]))*bar_width
        temp = [i for i,val in enumerate(bar_centers) if val>(bar_ctr-(ln_gc/2)*bar_width) and val<(bar_ctr+(ln_gc/2)*bar_width)]
        h = [0]*len(group_colors)
        for j in temp:
            h[group_id[j]] = bar_heights[j]
        group_heights+=[h]
        group_center+=[bar_ctr]
        bar_centers = np.delete(np.array(bar_centers),temp,axis=0)
        bar_heights = np.delete(np.array(bar_heights),temp)
        group_id = np.delete(np.array(group_id),temp)
    # print(group_colors,group_leg_labels,group_heights,group_center)

    ''' Map pixels to original coordinates'''
    Xlabel,Ylabel,xbox_centers,ybox_centers = get_text_labels(img,root)
    ybox_centers= np.array([(ybox_centers[i][0],abs(ybox_centers[i][1]-Y)) for i in range(len(ybox_centers))])
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
    group_heights = np.array(group_heights)
    first_label = Xlabel[(np.where(xbox_centers[:,0]==min(xbox_centers[:,0])))[0][0]]
    first_label -= (min(xbox_centers[:,0])-base_val)*normalize_scalex
    # group_heights = np.array(group_heights)*normalize_scalex + first_label
    group_heights = np.array(group_heights).astype(np.float32)
    for i in range(len(group_heights)):
        for j in range(len(h)):
            if group_heights[i][j] != 0:
                group_heights[i][j] = (group_heights[i][j]*normalize_scalex) + first_label
    if str(sorted(Ylabel)[0]).isnumeric():
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
        first_label = Ylabel[(np.where(ybox_centers[:,1] == min(ybox_centers[:,1])))[0][0]]
        group_center = np.array(group_center) - min(ybox_centers[:,1])
        group_center = np.array(group_center)*normalize_scaley + first_label
        bar_width = np.array(bar_width)*normalize_scaley
        labels = group_center
    else:
        i=0
        labels=[]
        while i<len(group_center):
            ids = [j for j,val in enumerate(np.array(ybox_centers)[:,1]) if (val>group_center[i]-(ln_gc/2)*bar_width and val<group_center[i]+(ln_gc/2)*bar_width)]
            if(len(ids)!=0):
                labels+=[' '.join([Ylabel[j] for j in ids])]
                ybox_centers = np.delete(ybox_centers,ids,axis=0)
                Ylabel = np.delete(Ylabel,ids)
            else:
                # IF for a given height there is no x label detected we add a default label
                labels+=['No Label']
            i+=1
        # IF there are still some labels left then the bar height is zero
        i=0
        group_heights=group_heights.tolist()
        while i<len(ybox_centers):
            group_center+=[int(ybox_centers[i][1])]
            labels+=[Ylabel[i]]
            group_heights+=[[0]*len(group_heights[0])]
            i+=1
        plt.yticks(group_center, labels, fontsize=10)
    # print(group_colors,group_leg_labels,group_heights,group_center)

    '''Reconstruct Grouped bar'''
    group_heights = np.array(group_heights)
    for i in range(len(group_colors)):
            k = ((len(group_colors)-1)/2-i)*bar_width
            plt.barh(np.array(group_center)+k,width=group_heights[:,i], height=bar_width, color=[np.array(group_colors[i][::-1])/255],edgecolor='k', label=group_leg_labels[i])
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.xlabel(get_xtitle(img,root))
    plt.ylabel(get_ytitle(img,root))
    plt.title(get_title(img,root))
    plt.tight_layout()
    plt.savefig(path+'reconstructed_'+image_name+".png")

    # Writing data to CSV file
    L = [['X']+group_leg_labels+['bar_width','title','x-title','y-title']]
    L = L + [[labels[0]]+group_heights[0].tolist()+[bar_width, get_title(img,root), get_xtitle(img,root), get_ytitle(img,root)]]
    L = L + [[labels[i]]+group_heights[i].tolist() for i in range(1,len(labels))]
    with open(path+'data_'+image_name+'.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(L)

def hist(filename):
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
    with open(path+'data_'+image_name+'.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(L)


    # plt.show()

def SH_bar(filename):
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

    #swap x and y co-ordinates and compute height same as for horizontal bars
    centers=np.array(centers)
    centers[:,[0, 1]] = centers[:,[1, 0]]
    centers = centers.tolist()
    unused_centers = sorted(centers)

    # stacked_bar
    '''compute all points belonging to base line and remove it from unused centers'''
    neigh_lmt=5
    #distances of centers from origin to find base point
    dist = list(np.array(unused_centers)[:,1]+np.array(unused_centers)[:,0])
    first_pt = unused_centers[dist.index(min(dist))]
    #distances of centers from origin to find base point
    dist = list(np.array(unused_centers)[:,1]+abs(X-np.array(unused_centers)[:,0]))
    last_pt = unused_centers[dist.index(min(dist))]
    base_val=(first_pt[1]+last_pt[1])//2
    heights = list(np.array(unused_centers)[:,1]-base_val)
    baseline_pts = [unused_centers[i] for i in range(len(heights)) if(heights[i]<neigh_lmt)]
    for i in baseline_pts:
        unused_centers.remove(i)

    '''find stacked heights form base pts'''
    i=0
    stack_heights=[]
    stacks_centers=[]
    totbar_height=[]
    while i<len(unused_centers):
        stacks=[] # to get heights of single stacked bar
        cntr=[]
        remove_ids=[]
        for j in range(i,len(unused_centers)):
            neigh_lmt=8
            if(abs(unused_centers[j][0]-unused_centers[i][0])<neigh_lmt):
                stacks+=[unused_centers[j][1]-base_val]
                cntr+=[[unused_centers[j][1],unused_centers[j][0]]]
                remove_ids+=[j]
        cntr, stacks= zip(*sorted(zip(cntr, stacks), reverse=True))
        cntr=np.array(cntr)
        cntr=[[np.mean(cntr[:,1]),(cntr[i][0]+cntr[i+1][0])/2] for i in range(len(cntr)-1)]+[[np.mean(cntr[:,1]),(cntr[len(cntr)-1][0]+base_val)/2]]
        # stacks=[stacks[i]-stacks[i+1] for i in range(len(stacks)-1)]+[stacks[len(stacks)-1]]
        stack_heights+=[stacks]
        stacks_centers+=[cntr]
        totbar_height+=[np.sum(stacks)]
        unused_centers = np.delete(unused_centers,remove_ids,axis=0)
    # print(totbar_height,"\n---------\n",stack_heights,"\n---------\n")
    # for I in stacks_centers:
    #     print(I)
    i=0
    neigh_lmt=7
    bar_height=[]
    bar_center=[]
    bar_width=[]
    while(i<len(totbar_height)):
        if((totbar_height[i]-totbar_height[i+1]<5) and (len(stack_heights[i])==len(stack_heights[i+1])) ):#neigh limit
            l = [abs(stack_heights[i][j]-stack_heights[i+1][j]) for j in range(len(stack_heights[i]))]
            # to belong to same stack
            if(max(l)<neigh_lmt):
                bar_height+=[((np.array(stack_heights[i])+np.array(stack_heights[i+1]))/2).tolist()]
                bar_center+=[((np.array(stacks_centers[i])+np.array(stacks_centers[i+1]))/2).tolist()]
                bar_width+=[abs(stacks_centers[i][0][0]-stacks_centers[i+1][0][0])//2]
                stack_heights = np.delete(stack_heights,i+1,axis=0)
                stacks_centers = np.delete(stacks_centers,i+1,axis=0)
                totbar_height.remove(totbar_height[i+1])
        else:
            temp1=[]
            temp2=[]
            if (len(stack_heights[i])< len(stack_heights[i+1])):
                s_id=i
                long_lst=list(stack_heights[i+1])
            else:
                s_id=i+1
                long_lst=list(stack_heights[i])
            for k in range(len(stack_heights[s_id])):
                id = [j for j,val in enumerate(long_lst) if abs(stack_heights[s_id][k]-val)<neigh_lmt]
                if(len(id)!=0):
                    del long_lst[id[0]]
                    temp1+=[stack_heights[s_id][k]]
                    temp2+=[[abs(stacks_centers[i][0][0]+stacks_centers[i+1][0][0])//2,stacks_centers[s_id][k][1]]]
            if(len(temp1)==len(stack_heights[s_id])):
                bar_height+=[temp1]
                bar_center+=[temp2]
                bar_width+=[abs(stacks_centers[i][0][0]-stacks_centers[i+1][0][0])//2]
                stack_heights = np.delete(stack_heights,s_id,axis=0)
                stacks_centers = np.delete(stacks_centers,s_id,axis=0)
                totbar_height.remove(totbar_height[s_id])
        i+=1
    stacks_centers=bar_center
    stack_heights=[]
    for stacks in bar_height:
        stack_heights+=[[stacks[i]-stacks[i+1] for i in range(len(stacks)-1)]+[stacks[len(stacks)-1]]]
    # stack_heights=bar_height
    bar_width = 2*np.mean(bar_width)
    # To make other heights as zeros
    max_len = np.max([len(a) for a in stack_heights])
    for i in range(len(stack_heights)):
        t=max_len-len(stack_heights[i])
        while(t>0):
            stack_heights[i]+=[0]
            stacks_centers[i]+=[[np.mean(np.array(stacks_centers[i])[:,0]),0]]
            t-=1
    stack_heights=np.array(stack_heights).astype(int)
    stacks_centers=np.array(stacks_centers).astype(int)
    # print(stacks_centers,stack_heights)

    ''' To get legend colors and its labels'''
    img = cv2.imread(path+image_name+".png")
    root = ET.parse(path+image_name+'.xml').getroot()
    # Now group stack heights based on it's catogery
    group_colors,group_leg_labels = get_legends(img,root)
    group_heights = []
    group_center = []
    for i in range(len(stacks_centers)):
        group_center += [int(np.mean([j[0] for j in stacks_centers[i]]))]
        l=[]
        for j in range(len(stacks_centers[i])):
            if(stack_heights[i][j]!=0):
                l+=[img[Y-stacks_centers[i][j][0],stacks_centers[i][j][1]].tolist()]
        h=[]
        for j in range(len(group_colors)):
            if group_colors[j] in l:
                id = l.index(group_colors[j])
                h += [stack_heights[i][id]]
            else :
                h += [0]
        group_heights += [h]
    # print(group_colors,group_leg_labels,group_heights,group_center)

    ''' Map pixels to original coordinates'''
    Xlabel,Ylabel,xbox_centers,ybox_centers = get_text_labels(img,root)
    ybox_centers= np.array([(ybox_centers[i][0],abs(ybox_centers[i][1]-Y)) for i in range(len(ybox_centers))])
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
    group_heights = np.array(group_heights)
    first_label=Xlabel[(np.where(xbox_centers[:,0]==min(xbox_centers[:,0])))[0][0]]
    first_label -= (min(xbox_centers[:,0])-base_val)*normalize_scalex
    group_heights = np.array(group_heights).astype(np.float32)
    for i in range(len(group_heights)):
        for j in range(len(h)):
            if group_heights[i][j] != 0:
                group_heights[i][j] = (group_heights[i][j]*normalize_scalex) + first_label

    if str(sorted(Ylabel)[0]).isnumeric():
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
        first_label = Ylabel[(np.where(ybox_centers[:,1] == min(ybox_centers[:,1])))[0][0]]
        group_center = np.array(group_center) - min(ybox_centers[:,1])
        group_center = np.array(group_center)*normalize_scaley + first_label
        bar_width = np.array(bar_width)*normalize_scaley
        labels = group_center
    else:
        i=0
        labels=[]
        while i<len(group_center):
            ids = [j for j,val in enumerate(np.array(ybox_centers)[:,1]) if (val>group_center[i]-bar_width//2 and val<group_center[i]+bar_width//2)]
            if(len(ids)!=0):
                labels+=[' '.join([Ylabel[j] for j in ids])]
                ybox_centers = np.delete(ybox_centers,ids,axis=0)
                Ylabel = np.delete(Ylabel,ids)
            else:
                # IF for a given height there is no x label detected we add a default label
                labels+=['No Label']
            i+=1
        # IF there are still some labels left then the bar height is zero
        i=0
        group_heights=group_heights.tolist()
        while i<len(ybox_centers):
            group_center+=[int(ybox_centers[i][1])]
            labels+=[Ylabel[i]]
            group_heights+=[[0]*len(group_heights[0])]
            i+=1
        plt.yticks(group_center, labels, fontsize=10)
    # print(group_colors,group_leg_labels,group_heights,group_center)

    '''Reconstruct bar'''
    group_heights = np.array(group_heights)
    for j in range(len(group_colors)-1,-1,-1):
        # for each stack of bar
        plt.barh(group_center,width=np.sum(group_heights[:,:j+1], axis = 1), height=bar_width, color=[np.array(group_colors[j][::-1])/255],edgecolor='k', label=group_leg_labels[j])
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.xlabel(get_xtitle(img,root))
    plt.ylabel(get_ytitle(img,root))
    plt.title(get_title(img,root))
    plt.tight_layout()
    plt.savefig(path+'reconstructed_'+image_name+".png")

    # Writing data to CSV file
    L = [['X']+group_leg_labels+['bar_width','title','x-title','y-title']]
    L = L + [[labels[0]]+group_heights[0].tolist()+[bar_width, get_title(img,root), get_xtitle(img,root), get_ytitle(img,root)]]
    L = L + [[labels[i]]+group_heights[i].tolist() for i in range(1,len(labels))]
    with open(path+'data_'+image_name+'.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(L)

def S_bar(filename):
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
    unused_centers = sorted(centers)
    # # plot data with seaborn
    # plt.scatter(cord_list["x_val"],  cord_list["y_val"], s=10, c='b')
    # plt.suptitle("Degenerate points and cluster centers")
    # plt.scatter(np.array(centers)[:,0], np.array(centers)[:,1], s=10, c='r', alpha=0.7)
    # plt.axis('off')

    # stacked_bar
    '''compute all points belonging to base line and remove it from unused centers'''
    neigh_lmt=6
    #distances of centers from origin to find base point
    dist = list(np.array(unused_centers)[:,1]+np.array(unused_centers)[:,0])
    first_pt = unused_centers[dist.index(min(dist))]
    #distances of centers from origin to find base point
    dist = list(np.array(unused_centers)[:,1]+abs(X-np.array(unused_centers)[:,0]))
    last_pt = unused_centers[dist.index(min(dist))]
    base_val=(first_pt[1]+last_pt[1])//2
    heights = list(np.array(unused_centers)[:,1]-base_val)
    baseline_pts = [unused_centers[i] for i in range(len(heights)) if(heights[i]<neigh_lmt)]
    for i in baseline_pts:
        unused_centers.remove(i)

    '''find stacked heights form base pts'''
    i=0
    stack_heights=[]
    stacks_centers=[]
    totbar_height=[]
    while i<len(unused_centers):
        stacks=[] # to get heights of single stacked bar
        cntr=[]
        remove_ids=[]
        for j in range(i,len(unused_centers)):
            if(abs(unused_centers[j][0]-unused_centers[i][0])<neigh_lmt):
                stacks+=[unused_centers[j][1]-base_val]
                cntr+=[[unused_centers[j][1],unused_centers[j][0]]]
                remove_ids+=[j]
        cntr, stacks= zip(*sorted(zip(cntr, stacks), reverse=True))
        cntr=np.array(cntr)
        cntr=[[np.mean(cntr[:,1]),(cntr[i][0]+cntr[i+1][0])/2] for i in range(len(cntr)-1)]+[[np.mean(cntr[:,1]),(cntr[len(cntr)-1][0]+base_val)/2]]
        # stacks=[stacks[i]-stacks[i+1] for i in range(len(stacks)-1)]+[stacks[len(stacks)-1]]
        stack_heights+=[stacks]
        stacks_centers+=[cntr]
        totbar_height+=[np.sum(stacks)]
        unused_centers = np.delete(unused_centers,remove_ids,axis=0)
    # print(totbar_height,"\n---------\n",stack_heights,"\n---------\n")
    # for I in stacks_centers:
    #     print(I)
    i=0
    neigh_lmt=7
    bar_height=[]
    bar_center=[]
    bar_width=[]
    while(i<len(totbar_height)):
        if((totbar_height[i]-totbar_height[i+1]<5) and (len(stack_heights[i])==len(stack_heights[i+1])) ):#neigh limit
            l = [abs(stack_heights[i][j]-stack_heights[i+1][j]) for j in range(len(stack_heights[i]))]
            # to belong to same stack
            if(max(l)<neigh_lmt):
                bar_height+=[((np.array(stack_heights[i])+np.array(stack_heights[i+1]))/2).tolist()]
                bar_center+=[((np.array(stacks_centers[i])+np.array(stacks_centers[i+1]))/2).tolist()]
                bar_width+=[abs(stacks_centers[i][0][0]-stacks_centers[i+1][0][0])//2]
                stack_heights = np.delete(stack_heights,i+1,axis=0)
                stacks_centers = np.delete(stacks_centers,i+1,axis=0)
                totbar_height.remove(totbar_height[i+1])
        else:
            temp1=[]
            temp2=[]
            if (len(stack_heights[i])< len(stack_heights[i+1])):
                s_id=i
                long_lst=list(stack_heights[i+1])
            else:
                s_id=i+1
                long_lst=list(stack_heights[i])
            for k in range(len(stack_heights[s_id])):
                id = [j for j,val in enumerate(long_lst) if abs(stack_heights[s_id][k]-val)<neigh_lmt]
                if(len(id)!=0):
                    del long_lst[id[0]]
                    temp1+=[stack_heights[s_id][k]]
                    temp2+=[[abs(stacks_centers[i][0][0]+stacks_centers[i+1][0][0])//2,stacks_centers[s_id][k][1]]]
            if(len(temp1)==len(stack_heights[s_id])):
                bar_height+=[temp1]
                bar_center+=[temp2]
                bar_width+=[abs(stacks_centers[i][0][0]-stacks_centers[i+1][0][0])//2]
                stack_heights = np.delete(stack_heights,s_id,axis=0)
                stacks_centers = np.delete(stacks_centers,s_id,axis=0)
                totbar_height.remove(totbar_height[s_id])
        i+=1
    stacks_centers=bar_center
    stack_heights=[]
    for stacks in bar_height:
        stack_heights+=[[stacks[i]-stacks[i+1] for i in range(len(stacks)-1)]+[stacks[len(stacks)-1]]]
    # stack_heights=bar_height
    bar_width = 2*np.mean(bar_width)
    # To make other heights as zeros
    max_len = np.max([len(a) for a in stack_heights])
    for i in range(len(stack_heights)):
        t=max_len-len(stack_heights[i])
        while(t>0):
            stack_heights[i]+=[0]
            stacks_centers[i]+=[[np.mean(np.array(stacks_centers[i])[:,0]),0]]
            t-=1
    stack_heights=np.array(stack_heights).astype(int)
    stacks_centers=np.array(stacks_centers).astype(int)
    # print(stacks_centers,stack_heights)

    ''' To get legend colors and its labels'''
    img = cv2.imread(path+image_name+".png")
    root = ET.parse(path+image_name+'.xml').getroot()
    # Now group stack heights based on it's catogery
    group_colors,group_leg_labels = get_legends(img,root)
    group_heights = []
    group_center = []
    for i in range(len(stacks_centers)):
        group_center += [int(np.mean([j[0] for j in stacks_centers[i]]))]
        l=[]
        for j in range(len(stacks_centers[i])):
            if(stack_heights[i][j]!=0):
                l+=[img[Y-stacks_centers[i][j][1],stacks_centers[i][j][0]].tolist()]

        h=[]
        for j in range(len(group_colors)):
            if group_colors[j] in l:
                id = l.index(group_colors[j])
                h += [stack_heights[i][id]]
            else :
                h += [0]
        group_heights += [h]
    # print(group_colors,group_leg_labels,group_heights,group_center)
    ''' Map pixels to original coordinates'''
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
    group_heights = np.array(group_heights)
    first_label = Ylabel[(np.where(ybox_centers[:,1] == max(ybox_centers[:,1])))[0][0]]
    first_label -= (Y-max(ybox_centers[:,1])-base_val)*normalize_scaley
    group_heights = np.array(group_heights).astype(np.float32)
    for i in range(len(group_heights)):
        for j in range(len(h)):
            if group_heights[i][j] != 0:
                group_heights[i][j] = (group_heights[i][j]*normalize_scaley) + first_label
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
        group_center = np.array(group_center) - min(xbox_centers[:,0])
        group_center = np.array(group_center)*normalize_scalex + first_label
        bar_width = np.array(bar_width)*normalize_scalex
        labels = group_center
    else:
        i=0
        labels=[]
        while i<len(group_center):
            ids = [j for j,val in enumerate(np.array(xbox_centers)[:,0]) if (val>group_center[i]-bar_width//2 and val<group_center[i]+bar_width//2)]
            if(len(ids)!=0):
                labels+=[' '.join([Xlabel[j] for j in ids])]
                xbox_centers = np.delete(xbox_centers,ids,axis=0)
                Xlabel = np.delete(Xlabel,ids)
            else:
                # IF for a given height there is no x label detected we add a default label
                labels+=[' No label ']
            i+=1
        # IF there are still some labels left then the bar height is zero
        i=0
        group_heights = group_heights.tolist()
        while i<len(xbox_centers):
            group_center+=[int(xbox_centers[i][0])]
            labels+=[Xlabel[i]]
            group_heights+=[[0]*len(group_heights[0])]
            i+=1
        plt.xticks(group_center, labels, rotation=90, fontsize=10)
    # print(group_colors,group_leg_labels,group_heights,group_center)


    '''Reconstruct Stacked bar'''
    group_heights = np.array(group_heights)
    for j in range(len(group_colors)-1,-1,-1):
        # for each stack of bar
        plt.bar(group_center,height=np.sum(group_heights[:,:j+1], axis = 1), width=bar_width, color=[np.array(group_colors[j][::-1])/255],edgecolor='k', label=group_leg_labels[j])
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.xlabel(get_xtitle(img,root))
    plt.ylabel(get_ytitle(img,root))
    plt.title(get_title(img,root))
    plt.tight_layout()
    plt.savefig(path+'reconstructed_'+image_name+".png")


    # Writing data to CSV file
    L = [['X']+group_leg_labels+['bar_width','title','x-title','y-title']]
    L = L + [[labels[0]]+group_heights[0].tolist()+[bar_width, get_title(img,root), get_xtitle(img,root), get_ytitle(img,root)]]
    L = L + [[labels[i]]+group_heights[i].tolist() for i in range(1,len(labels))]
    with open(path+'data_'+image_name+'.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(L)



