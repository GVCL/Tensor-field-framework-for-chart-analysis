import numpy as np
import cv2
from Computation_and_visualization_tool.Tensor_field_computation.Graph_Obj_Fill import color_fill

def segment(img,chart_type):
    # if alpha channel is having zero it represents no color
    if(img.shape[2]==4):
        img[img[:,:,3]==0] = [255,255,255,255]
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # l=list(np.ravel(thresh))
    # to create a threshold with white background
    # if(l.count(0)<(len(l)-l.count(0))):
    #     thresh = 255 - thresh
    #     img = 255 - img
    kernel = np.ones((3,3),np.uint8)
    l=list(np.ravel(thresh))
    if((1-(l.count(0)/len(l)))<=0.18 and chart_type!='scatter') :
        thresh = color_fill(img)
        print("Filled hollow objects......!")

    thresh= cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(thresh,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,5)
    l=list(np.ravel(dist_transform.astype(np.uint8)))
    dict={}
    for i in np.unique(l):
        dict.update({i:l.count(i)})
    t_min=(sorted(dict.items(), key=lambda x: (x[1],x[0]), reverse=True))[int(0.25*len(dict))][0]
    t_max=(sorted(dict.items(), key=lambda x: (x[1],x[0]), reverse=True))[int(0.5*len(dict))][0]
    [h,w] = dist_transform.shape
    # ret, thre1 = cv2.threshold(dist_transform,t_min,255,0)
    thre1=np.zeros((h,w), dtype=np.uint8)
    for i in range(h) :
        for j in range(w) :
            x=t_max
            a=i-2
            b=i+3
            c=j-2
            d=j+3
            if(i<2):
                a=0
            if(j<3):
                c=0
            if(i>h-2):
                b=h
            if(j>w-3):
                d=w
            if(np.amax(dist_transform[a:b,c:d])>t_min and np.amax(dist_transform[a:b,c:d])<t_max ):
                x=np.amax(dist_transform[a:b,c:d])
            if dist_transform[i,j] < x:
                thre1[i,j]=int(0)#black
            else:
                thre1[i,j]=int(255)#white
    # Finding unknown region
    sure_fg = thre1.astype(np.uint8)


    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    # img[markers == -1] = [255,0,0]
    img[markers == 1] = [255,255,255]

    return img

