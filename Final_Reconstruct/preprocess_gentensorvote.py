import cv2
import numpy as np
import xml.etree.ElementTree as ET
from Chart_Seg.Graph_Obj_Seg import segment
from compute_tensorvote import image_read

path = "/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Data/grouped_bar/gb02/"
# path = "/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Generated_data/H_stacked_bar/h_sb03/"
chart_type = 'bar'
image_name = path.split('/')
image_name = image_name[len(image_name)-2]

im = cv2.imread(path+image_name+".png")
h,w,_= np.shape(im)
seg_img = 255 - np.zeros((h,w,3), dtype=np.uint8)

# extract canvas
root = ET.parse(path+image_name+'.xml').getroot()
for obj in root.findall('object'):
    name = obj.find('name').text
    if name=='canvas':
        box = obj.find('bndbox')
        (x0,y0)=(int(box.find('xmin').text),int(box.find('ymin').text))
        (x1,y1)=(int(box.find('xmax').text),int(box.find('ymax').text))
        # remove gridlines by segmentation
        seg_img[y0:y1,x0:x1,:] = segment(im[y0:y1,x0:x1,:], chart_type)
        print("Segmentation Done.....!")

# im = cv2.GaussianBlur(seg_img, (5, 5), 1)
# gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# gray = cv2.equalizeHist(gray)
cedge_leftcorners = cv2.Canny(seg_img,100,200)
cedge_rightcorners = cv2.flip(cv2.Canny(cv2.flip(seg_img, 1),100,200),1)
cedge = cv2.bitwise_or(cedge_leftcorners, cedge_rightcorners, mask = None)
# # ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(cedge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
im = cv2.drawContours(seg_img, contours, -1, (0,0,0), 2)
print("Added Borders.....!")

filename=path+"canvas_"+image_name+".png"
cv2.imwrite(filename,im)
image_read(path,filename)

print("Preprocessed "+path)

cv2.waitKey(0)


