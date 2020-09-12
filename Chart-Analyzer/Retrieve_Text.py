import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from scipy.ndimage import rotate
from PIL import Image
from CRAFT_TextDetector import detect_text
from Deep_TextRecognition import text_recog

def remove_legend(img,root):
    #to remove text and legend
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'legend':
            box = obj.find('bndbox')
            (x0,y0)=(int(box.find('xmin').text),int(box.find('ymin').text))
            (x1,y1)=(int(box.find('xmax').text),int(box.find('ymax').text))
            img[y0-1:y1+1,x0-1:x1+1,:]= np.ones(img[y0-1:y1+1,x0-1:x1+1,:].shape, dtype=np.uint8)*255

def remove_text(img):
    detected_label,boxes,box_centers,slope = detect_text(img)
    label = text_recog()
    centers=[]
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        x0,x1,y0,y1=(poly[0][0],poly[1][0],poly[1][1],poly[2][1])
        if x0<0:
            x0=0
        if y0<0:
            y0=0
        centers+=[[(x0+x1)//2,(y0+y1)//2]]
        cv2.fillPoly(img, pts = [poly.reshape((-1, 1, 2))], color=(255, 255, 255, 255))
    return img

def get_text_labels(img,root):
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name=='x-labels':
            box = obj.find('bndbox')
            (X_x0,X_y0)=(int(box.find('xmin').text),int(box.find('ymin').text))
            (x1,y1)=(int(box.find('xmax').text),int(box.find('ymax').text))
            # To extend label boundaries by 10 in all directions for proper char identification
            xlabel_im=img[X_y0-1:y1+1,X_x0-1:x1+1,:]
        if name=='y-labels':
            box = obj.find('bndbox')
            (Y_x0,Y_y0)=(int(box.find('xmin').text),int(box.find('ymin').text))
            (x1,y1)=(int(box.find('xmax').text),int(box.find('ymax').text))
            # To extend label boundaries by 1 in all directions for proper char identification
            ylabel_im=img[Y_y0-1:y1+1,Y_x0-1:x1+1,:]
    #preprocess the image
    scale_factor_x=3
    scale_factor_y=3
    img = cv2.resize(xlabel_im, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    xlabel_im  = cv2.dilate(img, kernel, iterations=1)
    img = cv2.resize(ylabel_im, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    ylabel_im  = cv2.dilate(img, kernel, iterations=1)
    detected_xlabel,boxes,xbox_centers,slope = detect_text(cv2.cvtColor(xlabel_im,cv2.COLOR_GRAY2BGR))
    Xlabel = text_recog()
    if(len(Xlabel)!=0 and sorted(Xlabel)[0].isnumeric()):
        xbox_centers = [xbox_centers[i] for i in range(len(Xlabel)) if Xlabel[i].isnumeric()]
        Xlabel = [int(i) for i in Xlabel if i.isnumeric()]
    detected_ylabel,boxes,ybox_centers,slope = detect_text(cv2.cvtColor(ylabel_im,cv2.COLOR_GRAY2BGR))
    Ylabel = text_recog()
    if(len(Ylabel)!=0 and sorted(Ylabel)[0].isnumeric()):
        ybox_centers = [ybox_centers[i] for i in range(len(Ylabel)) if Ylabel[i].isnumeric()]
        Ylabel = [int(i) for i in Ylabel if i.isnumeric()]
    xbox_centers=np.array(xbox_centers)
    ybox_centers=np.array(ybox_centers)
    # In xbox_centers for same x value find mid point of both boxes in same line
    i=0
    while i<(len(xbox_centers)):
        l=[i for i, val in enumerate(abs(xbox_centers[:,0]-xbox_centers[i][0])) if val<15]
        if len(l)>1:
            Xlabel[l[0]]=' '.join([Xlabel[i] for i in l])
            xbox_centers[l[1]][1]=np.mean([xbox_centers[i][1] for i in l])
            Xlabel = np.delete(Xlabel, l[1:])
            xbox_centers = np.delete(xbox_centers, l[1:], axis=0) # for 2D array
        i+=1
    # In ybox_centers for same y value find mid point of both boxes in same line
    i=0
    while i<(len(ybox_centers)):
        l=[i for i, val in enumerate(abs(ybox_centers[:,1]-ybox_centers[i][1])) if val<15]
        if len(l)>1:
            Ylabel[l[0]]=' '.join([Ylabel[i] for i in l])
            ybox_centers[l[0]][0]=np.mean([ybox_centers[i][0] for i in l])
            Ylabel = np.delete(Ylabel, l[1:])
            ybox_centers = np.delete(ybox_centers, l[1:], axis=0) # for 2D array
        i+=1
    if(len(Xlabel)==0 or len(Ylabel)==0):
        print("Failed Recognizing text label ... !")
        exit(0)
    # To convert box_centers to original pixel coordinate system in image rather than cropped image
    xbox_centers[:,0]=(xbox_centers[:,0]/scale_factor_x)+X_x0-1
    xbox_centers[:,1]=(xbox_centers[:,1]/scale_factor_y)+X_y0-1
    ybox_centers[:,0]=(ybox_centers[:,0]/scale_factor_x)+Y_x0-1
    ybox_centers[:,1]=(ybox_centers[:,1]/scale_factor_y)+Y_y0-1

    return Xlabel,Ylabel,xbox_centers,ybox_centers

def get_title(img,root):
    flag = True
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name=='title':
            box = obj.find('bndbox')
            (x0,y0)=(int(box.find('xmin').text),int(box.find('ymin').text))
            (x1,y1)=(int(box.find('xmax').text),int(box.find('ymax').text))
            # To extend label boundaries by 10 in all directions for proper char identification
            title_im=img[y0-1:y1+1,x0-1:x1+1,:]
            flag = False
    if flag:
        return '_'
    #preprocess the image
    scale_factor_x=3
    scale_factor_y=3
    img = cv2.resize(title_im, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    title_im  = cv2.dilate(img, kernel, iterations=1)

    _,_,_,slope = detect_text(cv2.cvtColor(title_im,cv2.COLOR_GRAY2BGR))
    title_im = rotate(title_im, np.mean(slope), cval=255)


    # data extraction from the image
    title = pytesseract.image_to_data(title_im, output_type=Output.DICT)['text']
    title= ' '.join(list(filter(lambda a: a!= ' ' and a!= '', title)))

    return title

def get_xtitle(img,root):
    flag = True
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name=='x-title':
            box = obj.find('bndbox')
            (x0,y0)=(int(box.find('xmin').text),int(box.find('ymin').text))
            (x1,y1)=(int(box.find('xmax').text),int(box.find('ymax').text))
            # To extend label boundaries by 10 in all directions for proper char identification
            xtitle_im=img[y0-1:y1+1,x0-1:x1+1,:]
            flag = False
    if flag:
        return '_'
    #preprocess the image
    scale_factor_x=3
    scale_factor_y=3
    img = cv2.resize(xtitle_im, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    xtitle_im  = cv2.dilate(img, kernel, iterations=1)

    _,_,_,slope = detect_text(cv2.cvtColor(xtitle_im,cv2.COLOR_GRAY2BGR))
    xtitle_im = rotate(xtitle_im, np.mean(slope), cval=255)

    # data extraction from the image
    x = pytesseract.image_to_data(xtitle_im, output_type=Output.DICT)['text']
    x = ' '.join(list(filter(lambda a: a!= ' ' and a!= '', x)))

    return x

def get_ytitle(img,root):
    flag = True
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name=='y-title':
            box = obj.find('bndbox')
            (x0,y0)=(int(box.find('xmin').text),int(box.find('ymin').text))
            (x1,y1)=(int(box.find('xmax').text),int(box.find('ymax').text))
            ytitle_im= img[y0-1:y1+1,x0-1:x1+1,:]
            flag = False
    if flag:
        return '_'

    #preprocess the image
    scale_factor_x=3
    scale_factor_y=3
    img = cv2.resize(ytitle_im, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    ytitle_im  = cv2.dilate(img, kernel, iterations=1)

    _,_,_,slope = detect_text(cv2.cvtColor(ytitle_im,cv2.COLOR_GRAY2BGR))
    ytitle_im = rotate(ytitle_im, np.mean(slope), cval=255)


    # data extraction from the image
    y = pytesseract.image_to_data(ytitle_im, output_type=Output.DICT)['text']
    y = ' '.join(list(filter(lambda a: a!= ' ' and a!= '', y)))

    return y

def get_legends(img,root):
    flag = True
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'legend':
            box = obj.find('bndbox')
            (x0,y0)=(int(box.find('xmin').text),int(box.find('ymin').text))
            (x1,y1)=(int(box.find('xmax').text),int(box.find('ymax').text))
            # To extend label boundaries by 10 in all directions for proper char identification
            legend_im=img[y0-1:y1+1,x0-1:x1+1,:]
            flag = False
    if flag:
        return '_'

    #preprocess the image
    img = cv2.resize(legend_im, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    box_img,boxes,txtbox_center,slope = detect_text(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR))
    labels=text_recog()
   # rescale coordinates back
    boxes = (np.array(boxes)/3).tolist()
    txtbox_center = (np.array(txtbox_center)/3).tolist()
    # To remove text region from image, hoping background is white
    for i in boxes:
        [y0,y1,x0,x1]=[int(i[0][1]),int(i[2][1]),int(i[0][0]),int(i[1][0])]
        legend_im[y0:y1,x0:x1,:] = np.ones(np.shape(legend_im[y0:y1,x0:x1,:]), np.uint8) * 255
    # find contours in the thresholded image

    gray = cv2.equalizeHist(cv2.cvtColor(legend_im,cv2.COLOR_BGR2GRAY))
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    cedge_leftcorners = cv2.Canny(cv2.cvtColor(legend_im,cv2.COLOR_BGR2GRAY),100,200)
    cedge_rightcorners = cv2.flip(cv2.Canny(cv2.flip(cv2.cvtColor(legend_im,cv2.COLOR_BGR2GRAY), 1),100,200),1)
    cedge = cv2.bitwise_or(cedge_leftcorners, cedge_rightcorners, mask = None)
    contours, hierarchy = cv2.findContours(cedge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    legend_centers=[]
    for i in contours:
        # ignore contours of smaller area
        if cv2.contourArea(i)>10:
            legend_centers += np.mean(i,axis=0).astype(int).tolist()
    legend_colors = [legend_im[i[1],i[0]].tolist() for i in legend_centers]
    legend_centers, legend_colors = zip(*sorted(zip(legend_centers, legend_colors),reverse=True))
    legend_centers=np.array(legend_centers)
    legend_colors=np.array(legend_colors)
    # print(legend_centers,legend_colors)
    # cv2.drawContours(legend_im, contours, -1, (0,0,0), 2)
    # cv2.imshow("888",legend_im)
    # cv2.waitKey(0)

    legend_labels=[]
    i=0
    while(i<len(legend_centers)):
        vals = list(abs(np.array(txtbox_center)[:,1] - legend_centers[i][1]))
        # 6 is neighbour limit in y axis and in x axis get text after color box center
        l=np.array([(j,labels[j]) for j,txt in enumerate(txtbox_center) if (vals[j]<8 and (legend_centers[i][0]<txt[0])) ])
        if(len(l)!=0):
            legend_labels+=[' '.join(l[:,1])]
            txtbox_center = np.delete(txtbox_center,l[:,0].astype(int),axis=0)
            labels = np.delete(labels,l[:,0].astype(int))
        else:
            legend_centers = np.delete(legend_centers,i,axis=0)
            legend_colors = np.delete(legend_colors,i,axis=0)
            i-=1
        i+=1
    return legend_colors.tolist()[::-1],legend_labels[::-1]

