from PIL import Image
import pandas as pd
import numpy as np
import cv2
import csv
import xml.etree.ElementTree as ET

def write_image_to_csv(image):
    # Writing xy-coordinates and RGB values in csv
    with open("Image_RGB_test.csv", "w+") as my_csv:
        writer = csv.DictWriter(my_csv, fieldnames=["X", "Y", "Red", "Green", "Blue"])
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

def read_xml():
    tree = ET.parse('/home/komaldadhich/Desktop/scatter_2_1_lg.xml')
    root = tree.getroot()
    xmin =0,
    ymin =0,
    xmax =0,
    ymax =0
    # for obj in root.find('object'):
    #     print(obj.att)
    #     for b_box in obj.findall('bndbox'):
    #         xmin = b_box.find('xmin').text
    #         xmax = b_box.find('xmax').text
    #         ymin = b_box.find('ymin').text
    #         ymax = b_box.find('ymax').text
    # return xmin,ymin,xmax,ymax
    #
    for obj in root.iter('object'):
        for name_attr in obj.iter('name'):
            if name_attr.text == 'canvas':
                for rect in obj.iter('bndbox'):
                    xmin = rect.find('xmin').text
                    xmax = rect.find('xmax').text
                    ymin = rect.find('ymin').text
                    ymax = rect.find('ymax').text
    return xmin, ymin, xmax, ymax


def recreate_image(image,xmin,ymin,xmax,ymax):
    width, height, c = image.shape
    print(image.shape)
    blank_img = 255 * np.ones(shape=[width, height, 3], dtype=np.uint8)

    write_image_to_csv(image)
    df = pd.read_csv('Image_RGB_test.csv', sep=',', index_col=False)
    for i, row in df.iterrows():
        if row['X']>=int(xmin) and row['X']<=int(xmax) and row['Y']>=int(ymin) and row['Y']<=int(ymax):
            blank_img[row['Y'], row['X']]= [row['Red'], row['Green'], row['Blue']]

    cv2.imwrite("/home/komaldadhich/Desktop/scatter_2_1_lg._seg.png", blank_img)
    # cv2.imshow("White Blank", blank_img)
    # cv2.waitKey()

xmin,ymin,xmax,ymax = read_xml()
image = cv2.imread("/home/komaldadhich/Desktop/scatter_2_1_lg.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image2 = image[::-1, :, :]
# cv2.imshow("image", image)
# cv2.waitKey()

recreate_image(image,xmin,ymin,xmax,ymax)


# img = Image.new('RGB', (x,y), (255, 255, 255))
# img.save("image.png", "PNG")

