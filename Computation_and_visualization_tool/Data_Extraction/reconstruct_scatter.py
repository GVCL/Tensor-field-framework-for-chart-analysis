import xml.etree.ElementTree as ET
from Text_DET_REC.Retrieve_Text import get_text_labels,get_title,get_xtitle,get_ytitle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import *
from operator import add
import seaborn as sns
import csv
import cv2

path = "/Chart_Reconstruct/Generated_data/scatter/sc01/"
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


# Reconstruction of Scatter

X_val = np.array(sorted(centers))[:,0]
Y_val= np.array(sorted(centers))[:,1]

L = [['X','Y']]
L = L + [[X_val[i], Y_val[i]] for i in range(1,len(X_val))]
with open(path+'pix_data.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(L)

# img = cv2.imread(path+image_name+".png")
# root = ET.parse(path+image_name+'.xml').getroot()
# Xlabel,Ylabel,xbox_centers,ybox_centers = get_text_labels(img,root)
# if(isinstance(Ylabel[0], str) and Ylabel[0].isnumeric()):
#     Ylabel = [int(i) for i in Ylabel if i.isnumeric()]
# # To deal with duplicate values and make one as negative
# for i in np.unique(Ylabel):
#     id=[j for j, val in enumerate(Ylabel) if i==val]
#     if(len(id)==2):
#         if(ybox_centers[id[0]][1]>ybox_centers[id[1]][1]):
#             Ylabel[id[1]]*=-1
#             neg_ids=np.where(ybox_centers[:,1] < ybox_centers[id[1]][1])[0]
#         else:
#             Ylabel[id[0]]*=-1
#             neg_ids=np.where(ybox_centers[:,0] < ybox_centers[id[0]][1])[0]
#         for i in neg_ids:
#             Ylabel[i]*=-1
# normalize_scaley =abs((Ylabel[0]-Ylabel[1])/(ybox_centers[0][1]-ybox_centers[1][1]))
# Y_val = np.array(Y_val)
# first_label = Ylabel[(np.where(ybox_centers[:,1] == max(ybox_centers[:,1])))[0][0]]
# first_label -= (Y-max(ybox_centers[:,1]))*normalize_scaley
# Y_val = np.array(Y_val)*normalize_scaley + round(first_label)
# if str(sorted(Xlabel)[0]).isnumeric():
#     # To deal with duplicate values and make one as negative
#     for i in np.unique(Xlabel):
#         id=[j for j, val in enumerate(Xlabel) if i==val]
#         if(len(id)==2):
#             if(xbox_centers[id[0]][1]>xbox_centers[id[1]][1]):
#                 Xlabel[id[1]]*=-1
#                 neg_ids=np.where(xbox_centers[:,1] < xbox_centers[id[1]][1])[0]
#             else:
#                 Xlabel[id[0]]*=-1
#                 neg_ids=np.where(xbox_centers[:,0] < xbox_centers[id[0]][1])[0]
#             for i in neg_ids:
#                 Xlabel[i]*=-1
#
#     normalize_scalex = (Xlabel[0]-Xlabel[1])/(xbox_centers[0][0]-xbox_centers[1][0])
#     first_label=Xlabel[(np.where(xbox_centers[:,0]==min(xbox_centers[:,0])))[0][0]]
#     X_val = np.array(X_val) - min(xbox_centers[:,0])
#     X_val = np.array(X_val)*normalize_scalex + first_label
#     labels = X_val
# else:
#     i=0
#     labels=[]
#     while i<len(X_val):
#         ids = [j for j,val in enumerate(np.array(xbox_centers)[:,0]) if (val>X_val[i]-bar_width[0]//2 and val<X_val[i]+bar_width[0]//2)]
#         if(len(ids)!=0):
#             labels+=[' '.join([Xlabel[j] for j in ids])]
#             xbox_centers = np.delete(xbox_centers,ids,axis=0)
#             Xlabel = np.delete(Xlabel,ids)
#         else:
#             # IF for a given height there is no x label detected we add a default label
#             labels+=['NO LABEL']
#         i+=1
#     # IF there are still some labels left then the bar height is zero
#     i=0
#     Y_val=Y_val.tolist()
#     while i<len(xbox_centers):
#         X_val+=[int(xbox_centers[i][0])]
#         labels+=[Xlabel[i]]
#         Y_val+=[0]
#         i+=1
#     plt.xticks(X_val, labels, rotation=90, fontsize=10)
#
# # plt.bar(X,height=Y,width=bar_width, color=[[0.2,0.2,0.2]])
#
# # Reconstruct Scatter
# plt.plot(X_val, Y_val, 'ko')
# plt.xlabel(get_xtitle(img,root))
# plt.ylabel(get_ytitle(img,root))
# plt.title(get_title(img,root))
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig(path+'reconstructed_'+image_name+".png")
#
# # # Writing data to CSV file
# # L = [['X','Y','bar_width']]
# # L = L + [[labels[0], Y[0], bar_width]]
# # L = L + [[labels[i], Y[i]] for i in range(1,len(labels))]
# # with open(path+'data.csv', 'w', newline='') as file:
# #     writer = csv.writer(file, delimiter=',')
# #     writer.writerows(L)
#
# plt.show()
#
#
#
