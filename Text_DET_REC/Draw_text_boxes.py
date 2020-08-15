import pytesseract
from pytesseract import Output
# install tesseract along
#  brew install tesseract
# This formula contains only the "eng", "osd", and "snum" language data files.
# If you need any other supported languages, run `brew install tesseract-lang`.
import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from Text_DET_REC.CRAFT_TextDetector import detect_text
from Text_DET_REC.Deep_TextRecognition import text_recog

img = cv2.imread("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Data/histogram/hg-3/hg-3.png")
canvas = cv2.imread("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Data/histogram/hg-3/canvas.png")

#retrive the cropped image from annotated xml file
root = ET.parse("//Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Data/histogram/hg-3/hg-3.xml").getroot()

scale_factor_x=3
scale_factor_y=3

canvas = cv2.resize(canvas, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_CUBIC)
c1 = canvas.copy()
c2 = canvas.copy()

img = cv2.resize(img, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_CUBIC)
dup_img = img.copy()
dup_img2 = img.copy()
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kernel = np.ones((1, 1), np.uint8)
img = cv2.erode(img, kernel, iterations=1)
img  = cv2.dilate(img, kernel, iterations=1)
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
detected_label,boxes,box_centers=detect_text(img)
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
    cv2.polylines(dup_img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=5)
cv2.imwrite("text_detect.png",dup_img)

for i in range(len(label)):
    c1 = cv2.putText(c1, label[i], tuple(centers[i]), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6, cv2.LINE_AA)
cv2.imwrite("text_recog.png",c1)

# Tessract
d = pytesseract.image_to_data(img, output_type=Output.DICT)
'''left = x; top = y; right = x + width; bottom = y + height
    centroid of box is x+w/2,y+h/2 i.e left+width/2, top+height/2'''
text=[]
centers=[]
for i in range(len(d['text'])):
   if len(d['text'][i]) != 0 :
       text += [d['text'][i]]
       poly = np.array([[d['left'][i], d['top'][i]],
               [d['left'][i]+d['width'][i], d['top'][i]],
               [d['left'][i]+d['width'][i], d['top'][i]+d['height'][i]],
               [d['left'][i], d['top'][i]+d['height'][i]] ])
       x0,x1,y0,y1=(poly[0][0],poly[1][0],poly[1][1],poly[2][1])
       if x0<0:
           x0=0
       if y0<0:
           y0=0
       centers+=[[(x0+x1)//2,(y0+y1)//2]]
       cv2.polylines(dup_img2, [poly.reshape((-1, 1, 2))], True, color=(255, 0, 0), thickness=5)
cv2.imwrite("tessract_detect.png",dup_img2)
for i in range(len(text)):
    c2 = cv2.putText(c2, text[i], tuple(centers[i]), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 6, cv2.LINE_AA)
cv2.imwrite("tessract_recog.png",c2)

cv2.waitKey(0)
