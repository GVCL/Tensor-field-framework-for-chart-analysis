import pytesseract
from pytesseract import Output
import cv2
import os
import numpy as np
from scipy.ndimage import rotate
from PIL import Image
import xml.etree.ElementTree as ET
from Computation_and_visualization_tool.Text_DET_REC.CRAFT_TextDetector import detect_text
from Computation_and_visualization_tool.Text_DET_REC.Deep_TextRecognition import text_recog
filename="/Users/daggubatisirichandana/PycharmProjects/chart_percept/_Test/GroupedBar/gb3/gb03.png"
dup_img = cv2.imread(filename)
image_name = os.path.basename(filename).split(".png")[0]
path = os.path.dirname(filename)+'/'
# dup_img= img.copy

# dup_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# extract canvas
root = ET.parse(path+image_name+'.xml').getroot()
for obj in root.findall('object'):
    name = obj.find('name').text
    box = obj.find('bndbox')
    (x0,y0)=(int(box.find('xmin').text),int(box.find('ymin').text))
    (x1,y1)=(int(box.find('xmax').text),int(box.find('ymax').text))
    poly = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    cv2.polylines(dup_img, [poly.reshape((-1, 1, 2))], True, color=(0, 200, 0), thickness=2)
    if(name=='title'):
        image = cv2.putText(dup_img, name, (x0-35,y0+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2, cv2.LINE_AA)
    elif(name=='x-labels'):
        image = cv2.putText(dup_img, name, (x0-80,y0+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2, cv2.LINE_AA)
    elif(name=='y-labels'):
        image = cv2.putText(dup_img, name, (x0-30,y0+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2, cv2.LINE_AA)
    elif(name=='x-axis'):
            image = cv2.putText(dup_img, name, (x0+40,y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2, cv2.LINE_AA)
    elif(name=='y-axis'):
            image = cv2.putText(dup_img, name, (x0-25,y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2, cv2.LINE_AA)
    elif(name=='y-title'):
            image = cv2.putText(dup_img, name, (x0-5,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2, cv2.LINE_AA)
    elif(name=='canvas'):
            image = cv2.putText(dup_img, name, (x0+20,y0+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2, cv2.LINE_AA)
    else:
        image = cv2.putText(dup_img, name, (x0+10,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2, cv2.LINE_AA)
cv2.imwrite("/Users/daggubatisirichandana/PycharmProjects/chart_percept/_Test/annot_img.png",dup_img)
cv2.waitKey(0)
