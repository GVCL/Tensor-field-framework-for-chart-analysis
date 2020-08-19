#!/usr/bin/env python
#  -*- coding: utf-8 -*-

import cv2
import os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import xml.etree.ElementTree as ET
from generate_structure_tensor_lab import compute_structure_tensor
from data_extract import reconstruct_chart
# from generate_structure_tensor import compute_structure_tensor
from Graph_Obj_Seg import segment

# Add border to chart objects
def add_border(seg_img):
    cedge_leftcorners = cv2.Canny(seg_img,100,200)
    cedge_rightcorners = cv2.flip(cv2.Canny(cv2.flip(seg_img, 1),100,200),1)
    cedge = cv2.bitwise_or(cedge_leftcorners, cedge_rightcorners, mask = None)
    contours, hierarchy = cv2.findContours(cedge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im = cv2.drawContours(seg_img, contours, -1, (0,0,0), 2)
    # print("Added Borders.....!")

    return im

# Read image and compute tensor field
def image_read(filename):
    image_name = os.path.basename(filename).split(".png")[0]
    path = os.path.dirname(filename)+'/'
    im = Image.open(filename)
    # tkimage = ImageTk.PhotoImage(image=im)
    img = cv2.imread(filename)
    #From chart classification module
    chart_type='bar'

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
            # remove gridlines by segmentation
            seg_img[y0:y1,x0:x1,:] = segment(img[y0:y1,x0:x1,:], chart_type)
            # print("Segmentation Done.....!")
    image = add_border(seg_img)
    cv2.imwrite(path+"canvas_"+image_name+".png",image)
    print("Preprocessed "+filename)
    # image_clr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_clr = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image_smth = cv2.GaussianBlur(image_clr, (3, 3), 1)
    image2 = image_smth[::-1, :, :]
    data = write_image_to_csv(image2, path)
    compute_structure_tensor(data, image2, path)  # compute structure tensor
    print("Image Size", image_smth.shape)
    reconstruct_chart(filename,chart_type)

    im = Image.open(path+"reconstructed_"+image_name+".png")
    tkimage = ImageTk.PhotoImage(image=im)
    return image2, tkimage, im

# Writing xy-coordinates and RGB values of image in csv
def write_image_to_csv(image, path):
    with open(path+"Image_RGB.csv", "w+") as my_csv:
        writer = csv.DictWriter(my_csv, fieldnames=["X", "Y", "Red", "Green", "Blue"])
        writer.writeheader()
        y = 0
        for row in image:
            x = 0
            for col in row:
                my_csv.write("%d, %d, " % (x, y))
                for val in col:
                    my_csv.write("%d," % val)
                my_csv.write("\n")
                x += 1
            y += 1
    data = pd.read_csv(path+"Image_RGB.csv", sep=",", index_col=False)
    print("image file generated.")
    return data