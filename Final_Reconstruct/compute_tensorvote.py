import cv2
import csv
import pandas as pd
from Computation_visualization_tool.Tensor_field_computation.generate_structure_tensor_lab import compute_structure_tensor_path

def image_read(path,filename):
    img = cv2.imread(filename)
    image_clr = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    image_smth = cv2.GaussianBlur(image_clr, (3, 3), 1)
    image2 = image_smth[::-1, :, :]
    data = write_image_to_csv(path, image2)
    compute_structure_tensor_path(path, data, image2)  # compute structure tensor

# Writing xy-coordinates and RGB values of image in csv
def write_image_to_csv(path,image):
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
