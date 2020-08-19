#!/usr/bin/env python
#  -*- coding: utf-8 -*-

import cv2
import csv
from PIL import Image, ImageTk
from skimage import color
import pandas as pd
import numpy as np
from generate_structure_tensor_lab import compute_structure_tensor

def image_read(filename):
    file1 = open("myfile.txt", "w")
    path = [filename]
    file1.write(path)
    file1.close()
    image = cv2.imread(filename)
    image = image.astype("float32")/255
    print(image)
    image = image.astype("float32")
    print(image.dtype, image)
    image3 = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    print("lab", image3.dtype)

    image2 = cv2.GaussianBlur(image3, (3, 3), 1)
    print (image3.shape)
    # im = Image.fromarray(image3)
    # tkimage = ImageTk.PhotoImage(image=im)
    image2 = image2[::-1, :, :]
    data = write_image_to_csv(image2)
    compute_structure_tensor(data, image2)  # compute structure tensor
    # return image2, im, data
    return image2, data

def write_image_to_csv(image):
    # Writing xy-coordinates and RGB values in csv
    with open("Image_Lab.csv", "w+") as my_csv:
        writer = csv.DictWriter(my_csv, fieldnames=["X", "Y", "L*", "a*", "b*"])
        # writer = csv.DictWriter(my_csv, fieldnames=["X", "Y", "L", "A", "B"])
        writer.writeheader()
        y = 0
        for row in image:
            x = 0
            for col in row:
                my_csv.write("%d, %d, " % (x, y))
                for val in col:
                    my_csv.write("%d," % (val))

                my_csv.write("\n")
                x += 1
            y += 1

    data = pd.read_csv("Image_Lab.csv", sep=",", index_col=False)
    print("image file generated.")
    return data
