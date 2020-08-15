import xml.etree.ElementTree as ET
from Text_DET_REC.Retrieve_Text import get_text_labels,get_legends,get_xtitle,get_ytitle,get_title
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import *
from operator import add
import seaborn as sns
import csv
import cv2

path = "/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Generated_data/grouped_bar/gb01/"
image_name = path.split('/')
image_name = image_name[len(image_name)-2]
data_tensors = pd.read_csv(path+"tensor_vote_matrix.csv", sep=",", index_col=False)

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
baseline_pts = [unused_centers[i] for i in range(len(heights)) if(heights[i]<neigh_lmt)]
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
print(group_colors,group_leg_labels,group_heights,group_center)


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
with open(path+'data.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(L)

plt.show()
