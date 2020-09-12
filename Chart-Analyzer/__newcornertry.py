import cv2
import numpy as np
from sklearn.cluster import *
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from Graph_Obj_Seg import segment
from Chart_Classification.model_loader import classifyImage
from Retrieve_Text import *
import csv
import os

filename = "/Users/daggubatisirichandana/PycharmProjects/chart_percept/sampleplots/8032.png"
image_name = os.path.basename(filename).split(".png")[0]
path = os.path.dirname(filename)+'/'
img = cv2.imread(filename)
chart_type=classifyImage(filename)
h,w,_= np.shape(img)
seg_img = 255 - np.zeros((h,w,3), dtype=np.uint8)
# extract canvas
root = ET.parse(path+image_name+'.xml').getroot()
for obj in root.findall('object'):
    name = obj.find('name').text
    if name=='canvas':
        box = obj.find('bndbox')
        (x0,y0)=(int(box.find('xmin').text),int(box.find('ymin').text))
        (x1,y1)=(int(box.find('xmax').text),int(box.find('ymax').text))
        # remove text & gridlines by segmentation
        # seg_img[y0:y1,x0:x1,:] = segment(remove_text(img[y0:y1,x0:x1,:]), chart_type)
        seg_img[y0:y1,x0:x1,:] = segment(img[y0:y1,x0:x1,:], chart_type)
remove_legend(seg_img,root)

g_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
cedge_leftcorners = cv2.Canny(g_img,100,200)
cedge_rightcorners = cv2.flip(cv2.Canny(cv2.flip(g_img, 1),100,200),1)
cedge = cv2.bitwise_or(cedge_leftcorners, cedge_rightcorners, mask = None)
contours, hierarchy = cv2.findContours(cedge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# im = cv2.drawContours(seg_img, contours, -1, (0,0,0), 2)

data=[]
for poly in contours:
    data += poly.tolist()
data = np.array(data)[:,0]
db = DBSCAN(eps=4, min_samples=3).fit(data)
labels = db.labels_
centers=[]
for i in (np.unique(labels)[1:]):
    indexes = [id for id in range(len(labels)) if labels[id] == i]
    x=0
    y=0
    for k in indexes:
        x+=data[:,0][k]
        y+=data[:,1][k]
    centers+=[[x//len(indexes),y//len(indexes)]]
# print(centers)
unused_centers = sorted(centers)
X=h
Y=w

# # plot data with seaborn
# plt.imshow(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB),alpha=0.3)
# plt.scatter(data[:,0],  data[:,1], s=10, c='b')
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
plt.savefig(path+'R_'+image_name+".png")

# Writing data to CSV file
L = [['X']+group_leg_labels+['bar_width','title','x-title','y-title']]
L = L + [[labels[0]]+group_heights[0].tolist()+[bar_width, get_title(img,root), get_xtitle(img,root), get_ytitle(img,root)]]
L = L + [[labels[i]]+group_heights[i].tolist() for i in range(1,len(labels))]
with open(path+'D_'+image_name+'.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(L)

plt.show()
