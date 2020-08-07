#!/usr/bin/env python
#  -*- coding: utf-8 -*-

import cv2
import csv
from PIL import Image, ImageTk
import pandas as pd
from generate_structure_tensor_lab import compute_structure_tensor
# from generate_structure_tensor import compute_structure_tensor
from Graph_Obj_Seg import segment_img

# Add border to chart objects
def add_border(im):
    # im = cv2.imread(filename)
    im = cv2.GaussianBlur(im, (3, 3), 1)
    cedge = cv2.Canny(im, 100, 200)
    contours, hierarchy = cv2.findContours(cedge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im = cv2.drawContours(im, contours, -1, (0, 0, 0), 2)
    return im

# Read image and compute tensor field
def image_read(filename):
    im = Image.open(filename)
    tkimage = ImageTk.PhotoImage(image=im)
    img = cv2.imread(filename)
    img_seg = segment_img(img)
    image = add_border(img_seg)
    # image_clr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_clr = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image_smth = cv2.GaussianBlur(image_clr, (3, 3), 1)
    image2 = image_smth[::-1, :, :]
    data = write_image_to_csv(image2)
    print("st")
    compute_structure_tensor(data, image2)  # compute structure tensor
    print("Image Size", image_smth.shape)
    return image2, tkimage, im

# Writing xy-coordinates and RGB values of image in csv
def write_image_to_csv(image):
    with open("Image_RGB.csv", "w+") as my_csv:
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
    data = pd.read_csv("Image_RGB.csv", sep=",", index_col=False)
    print("image file generated.")
    return data
