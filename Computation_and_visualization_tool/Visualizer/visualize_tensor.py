from colormap import *
import matplotlib.colors as mpl
from tkinter import filedialog
import pandas as pd
import cv2
import seaborn as sns
import numpy as np
from matplotlib.patches import Ellipse
from math import atan, degrees
from PIL import ImageTk, Image
import matplotlib.cm as cm
from operator import add

def read_image(filename):
        image_org = cv2.imread(filename)
        image = image_org.astype("float32") / 255
        print(image)
        image = image.astype("float32")
        print(image.dtype, image)
        image3 = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        print("lab", image3.dtype)

        image2 = cv2.GaussianBlur(image3, (3, 3), 1)
        print(image3.shape)
        im = Image.open(filename)
        tkimage = ImageTk.PhotoImage(image=im)
        image2 = image_org[::-1, :, :]
        return image2, tkimage, im

# Visualise structured tensor
def visualize_tensor():
    csv_file_path = filedialog.askopenfilename(filetypes=[("csv File", ".csv")])
    data_tensors = pd.read_csv(csv_file_path, sep=",", index_col=False)
    X = len(data_tensors["X"].unique())
    Y = len(data_tensors["Y"].unique())

    cord_list = {
        "x_val": [],
        "y_val": [],
        "e1_vec": [],
        "e2_vec": [],
        "e3_vec": [],
        "e4_vec": [],
        "val1": [],
        "val2": [],
        "temp": [],
        "CL": [],
        "CP": [],
    }

    for i in range(X * Y):
        if data_tensors["val1"][i] != 0.0 or data_tensors["val2"][i] != 0.0:
            cord_list["x_val"].append(data_tensors["X"][i])
            cord_list["y_val"].append(data_tensors["Y"][i])
            cord_list["e1_vec"].append(data_tensors["val1"][i] * data_tensors["vec00"][i])
            cord_list["e2_vec"].append(data_tensors["val1"][i] * data_tensors["vec01"][i])
            cord_list["e3_vec"].append(data_tensors["val2"][i] * data_tensors["vec10"][i])
            cord_list["e4_vec"].append(data_tensors["val2"][i] * data_tensors["vec11"][i])
            cord_list["val1"].append((data_tensors["val1"][i]))
            cord_list["val2"].append((data_tensors["val2"][i]))
            cord_list["CL"].append((data_tensors["CL"][i]))
            cord_list["CP"].append((data_tensors["CP"][i]))

    e1 = [i * -1 for i in cord_list["e1_vec"]]
    e2 = [j * -1 for j in cord_list["e2_vec"]]
    e3 = [j * -1 for j in cord_list["e3_vec"]]
    e4 = [j * -1 for j in cord_list["e4_vec"]]

    fig, ax = plt.subplots()
    plt.suptitle("Structure tensor")
    # plt.grid(color="grey", linestyle="--", linewidth=0.5)
    norm = mpl.Normalize(vmin=0, vmax=1, clip=False)

    P = ax.quiver(
        cord_list["x_val"][::2],
        cord_list["y_val"][::2],
        e1[::2],
        e2[::2],
        norm=norm,
        headlength = 0,
        angles="uv",
        scale=5,
        scale_units="xy",
        linewidth=2,
        pivot='mid', color ='r'
    )
    Q = ax.quiver(
        cord_list["x_val"][::2],
        cord_list["y_val"][::2],
        cord_list["e1_vec"][::2],
        cord_list["e2_vec"][::2],
        norm=norm,
        headlength = 0,
        angles="uv",
        scale=5,
        scale_units="xy",
        linewidth=2,
        pivot='mid', color ='r'
    )
    R = ax.quiver(
        cord_list["x_val"][::2],
        cord_list["y_val"][::2],
        cord_list["e3_vec"][::2],
        cord_list["e4_vec"][::2],
        norm=norm,
        headlength = 0,
        angles="uv",
        scale=5,
        scale_units="xy",
        linewidth=2,
        pivot ='mid', color='b'
    )
    S = ax.quiver(
        cord_list["x_val"][::2],
        cord_list["y_val"][::2],
        e3[::2],
        e4[::2],
        cmap=colorMap_b(),
        norm=norm,
        headlength = 0,
        angles="uv",
        scale=5,
        scale_units="xy",
        linewidth=2,
        pivot='mid', color ='b'
    )

    plt.show()

#Visualize tensor voting before AD
def visualize_tensor_voting():
    csv_file_path = filedialog.askopenfilename(filetypes=[("csv File", ".csv")])
    data_tensors = pd.read_csv(csv_file_path, sep=",", index_col=False)
    X = len(data_tensors["X"].unique())
    Y = len(data_tensors["Y"].unique())

    cord_list = {
        "x_val": [],
        "y_val": [],
        "e1_vec": [],
        "e2_vec": [],
        "e3_vec": [],
        "e4_vec": [],
        "val1": [],
        "val2": [],
        "CL": [],
        "CP": []
    }

    for i in range(X * Y):

        if data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0:
            cord_list["x_val"].append(data_tensors["X"][i])
            cord_list["y_val"].append(data_tensors["Y"][i])
            cord_list["val1"].append(data_tensors["val1"][i])
            cord_list["val2"].append(data_tensors["val2"][i])
            cord_list["e1_vec"].append(data_tensors["val1"][i] * data_tensors["vec00"][i])
            cord_list["e2_vec"].append(data_tensors["val1"][i] * data_tensors["vec01"][i])
            cord_list["e3_vec"].append(data_tensors["val2"][i] * data_tensors["vec10"][i])
            cord_list["e4_vec"].append(data_tensors["val2"][i] * data_tensors["vec11"][i])
            cord_list["CL"].append((data_tensors["CL"][i]))
            cord_list["CP"].append((data_tensors["CP"][i]))

    print ("Visualizing Tensor Voting")
    e1 = [i * -1 for i in cord_list["e1_vec"]]
    e2 = [j * -1 for j in cord_list["e2_vec"]]
    e3 = [j * -1 for j in cord_list["e3_vec"]]
    e4 = [j * -1 for j in cord_list["e4_vec"]]

    fig, ax = plt.subplots()
    plt.suptitle("Tensor Voting before Anisotropic Diffusion")
    # plt.grid(color="grey", linestyle="--", linewidth=0.5)
    norm = mpl.Normalize(vmin=0, vmax=1, clip=False)
    P = ax.quiver(
        cord_list["x_val"][::2],
        cord_list["y_val"][::2],
        e1[::2],
        e2[::2],
        norm=norm,
        headlength = 0,
        angles="uv",
        scale=6,
        scale_units="xy",
        linewidth=2,
        pivot='mid', color ='r'
    )
    Q = ax.quiver(
        cord_list["x_val"][::2],
        cord_list["y_val"][::2],
        cord_list["e1_vec"][::2],
        cord_list["e2_vec"][::2],
        norm=norm,
        headlength = 0,
        angles="uv",
        scale=6,
        scale_units="xy",
        linewidth=2,
        pivot='mid', color ='r'
    )
    R = ax.quiver(
        cord_list["x_val"][::2],
        cord_list["y_val"][::2],
        cord_list["e3_vec"][::2],
        cord_list["e4_vec"][::2],
        norm=norm,
        headlength = 0,
        angles="uv",
        scale=6,
        scale_units="xy",
        linewidth=2,
        pivot ='mid', color='b'
    )
    S = ax.quiver(
        cord_list["x_val"][::2],
        cord_list["y_val"][::2],
        e3[::2],
        e4[::2],
        cmap=colorMap_b(),
        norm=norm,
        headlength = 0,
        angles="uv",
        scale=6,
        scale_units="xy",
        linewidth=2,
        pivot='mid', color ='b'
    )

    plt.show()

    # Visualize tensor voting post AD

#Visualize tensor voting after AD
def visualize_tensor_voting_AD():
    csv_file_path = filedialog.askopenfilename(filetypes=[("csv File", ".csv")])
    data_tensors = pd.read_csv(csv_file_path, sep=",", index_col=False)
    X = len(data_tensors["X"].unique())
    Y = len(data_tensors["Y"].unique())

    cord_list = {
        "x_val": [],
        "y_val": [],
        "e1_vec": [],
        "e2_vec": [],
        "e3_vec": [],
        "e4_vec": [],
        "ani_val1": [],
        "ani_val2": [],
        "CL": [],
        "CP": []
    }

    for i in range(X * Y):

        if data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0:
            cord_list["x_val"].append(data_tensors["X"][i])
            cord_list["y_val"].append(data_tensors["Y"][i])
            cord_list["ani_val1"].append((data_tensors["ani_val1"][i]))
            cord_list["ani_val2"].append((data_tensors["ani_val2"][i]))
            cord_list["e1_vec"].append(data_tensors["ani_val2"][i] * data_tensors["vec00"][i])
            cord_list["e2_vec"].append(data_tensors["ani_val2"][i] * data_tensors["vec01"][i])
            cord_list["e3_vec"].append(data_tensors["ani_val1"][i] * data_tensors["vec10"][i])
            cord_list["e4_vec"].append(data_tensors["ani_val1"][i] * data_tensors["vec11"][i])
            cord_list["CL"].append((data_tensors["CL"][i]))
            cord_list["CP"].append((data_tensors["CP"][i]))

    print ("Visualizing Tensor Voting")
    e1 = [i * -1 for i in cord_list["e1_vec"]]
    e2 = [j * -1 for j in cord_list["e2_vec"]]
    e3 = [j * -1 for j in cord_list["e3_vec"]]
    e4 = [j * -1 for j in cord_list["e4_vec"]]

    fig, ax = plt.subplots()
    plt.suptitle("Tensor Voting after Anisotropic Diffusion")
    # plt.grid(color="grey", linestyle="--", linewidth=0.5)
    norm = mpl.Normalize(vmin=0, vmax=1, clip=False)

    P = ax.quiver(
        cord_list["x_val"][::2],
        cord_list["y_val"][::2],
        e1[::2],
        e2[::2],
        norm=norm,
        headlength = 0,
        angles="uv",
        scale=1,
        scale_units="xy",
        linewidth=2,
        pivot='mid', color ='b'
    )
    Q = ax.quiver(
        cord_list["x_val"][::2],
        cord_list["y_val"][::2],
        cord_list["e1_vec"][::2],
        cord_list["e2_vec"][::2],
        norm=norm,
        headlength = 0,
        angles="uv",
        scale=1,
        scale_units="xy",
        linewidth=2,
        pivot='mid', color ='b'
    )
    R = ax.quiver(
        cord_list["x_val"][::2],
        cord_list["y_val"][::2],
        cord_list["e3_vec"][::2],
        cord_list["e4_vec"][::2],
        norm=norm,
        headlength = 0,
        angles="uv",
        scale=1,
        scale_units="xy",
        linewidth=2,
        pivot ='mid', color='r'
    )
    S = ax.quiver(
        cord_list["x_val"][::2],
        cord_list["y_val"][::2],
        e3[::2],
        e4[::2],
        cmap=colorMap_b(),
        norm=norm,
        headlength = 0,
        angles="uv",
        scale=1,
        scale_units="xy",
        linewidth=2,
        pivot='mid', color ='r'
    )

    plt.show()

#Visualize Saliency
def visualize_colormap(image):
    csv_file_path = filedialog.askopenfilename(filetypes=[("csv File", ".csv")])
    data_tensors = pd.read_csv(csv_file_path, sep=",", index_col=False)
    X = len(data_tensors["X"].unique())
    Y = len(data_tensors["Y"].unique())

    c_map = plt.cm.get_cmap('coolwarm', 100)
    cord_list = {
        "x_val": [],
        "y_val": [],
        "cord_val": [],
        "e1_vec": [],
        "e2_vec": [],
        "e3_vec": [],
        "e4_vec": [],
        "val1": [],
        "val2": [],
        "ani_val1": [],
        "ani_val2": [],
        "CL": [],
        "CP": [],
        "temp": [],
        "entropy": []
    }

    for i in range(X * Y):

        if data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0:
            cord_list["x_val"].append(data_tensors["X"][i])
            cord_list["y_val"].append(data_tensors["Y"][i])
            cord_list["CL"].append((data_tensors["CL"][i]))
            cord_list["CP"].append((data_tensors["CP"][i]))

    fig, ax = plt.subplots()
    plt.suptitle("Saliency Visualization")
    ax.axis('off')
    Q1 = plt.scatter(cord_list["x_val"],
                     cord_list["y_val"], s=10, c=cord_list['CL'], cmap=c_map)
    plt.colorbar(Q1)

    plt.show()

def visualize_distribution():
    csv_file_path = filedialog.askopenfilename(filetypes=[("csv File", ".csv")])
    data_tensors = pd.read_csv(csv_file_path, sep=",", index_col=False)
    X = len(data_tensors["X"].unique())
    Y = len(data_tensors["Y"].unique())

    cord_list = {
        "x_val": [],
        "y_val": [],
        "cord_val": [],
        "e1_vec": [],
        "e2_vec": [],
        "e3_vec": [],
        "e4_vec": [],
        "val1": [],
        "val2": [],
        "ani_val1": [],
        "ani_val2": [],
        "CL": [],
        "CP": [],
        "temp": [],
        "entropy": []
    }

    for i in range(X * Y):

        if data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0:
            cord_list["x_val"].append(data_tensors["X"][i])
            cord_list["y_val"].append(data_tensors["Y"][i])
            cord_list["CL"].append((data_tensors["CL"][i]))
            cord_list["CP"].append((data_tensors["CP"][i]))

    fig, ax = plt.subplots()
    plt.suptitle("Saliency Distribution")

    sns.distplot(cord_list['CP'], hist=False)
    plt.show()

def visualize_tv_ellipse(image):
    csv_file_path = filedialog.askopenfilename(filetypes=[("csv File", ".csv")])
    data_tensors = pd.read_csv(csv_file_path, sep=",", index_col=False)
    X = len(data_tensors["X"].unique())
    Y = len(data_tensors["Y"].unique())

    cord_list = {
        "x_val": [],
        "y_val": [],
        "e1_vec": [],
        "e2_vec": [],
        "e3_vec": [],
        "e4_vec": [],
        "ani_val1": [],
        "ani_val2": [],
        "val1": [],
        "val2": [],
        "CL": [],
        "CP": []
    }

    for i in range(X * Y):

        if data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0:
            cord_list["x_val"].append(data_tensors["X"][i])
            cord_list["y_val"].append(data_tensors["Y"][i])
            cord_list["ani_val1"].append((data_tensors["ani_val1"][i]))
            cord_list["ani_val2"].append((data_tensors["ani_val2"][i]))
            cord_list["val1"].append((data_tensors["val1"][i]))
            cord_list["val2"].append((data_tensors["val2"][i]))
            cord_list["e1_vec"].append(data_tensors["ani_val2"][i] * data_tensors["vec00"][i])
            cord_list["e2_vec"].append(data_tensors["ani_val2"][i] * data_tensors["vec01"][i])
            cord_list["e3_vec"].append(data_tensors["ani_val1"][i] * data_tensors["vec10"][i])
            cord_list["e4_vec"].append(data_tensors["ani_val1"][i] * data_tensors["vec11"][i])
            cord_list["CL"].append((data_tensors["CL"][i]))
            cord_list["CP"].append((data_tensors["CP"][i]))

    print("Visualizing Tensor Voting")
    xy_cord = zip(cord_list["x_val"], cord_list["y_val"])
    ells=[]
    count =0
    c_map = plt.cm.get_cmap('coolwarm', 100)(np.array(cord_list["CL"]))
    for t in xy_cord:
        angle_val = degrees(atan(cord_list["e4_vec"][count]/cord_list["e3_vec"][count]))
        if cord_list["ani_val1"][count]<0.1:
            temp = Ellipse(xy=t,
                            width=0.3, height=0.3,
                            angle=angle_val, color=c_map[count])
        else:
            temp = Ellipse(xy=t,
               width=cord_list["ani_val1"][count], height=cord_list["ani_val2"][count],
               angle=angle_val, color=c_map[count])

        ells.append(temp)
        count+=1
    norm = mpl.Normalize(vmin=0, vmax=1)
    fig, ax = plt.subplots()
    plt.suptitle("Tensor Voting")
    # print("image",image)
    for e in ells[::2]:
        ax.add_patch(e)

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('coolwarm', 100)))
    ax.set_xlim(0, max(cord_list["x_val"]))
    ax.set_ylim(0, max(cord_list["y_val"]))
    ax.axis('off')

    if image is not None:
        im_alpha = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        for i in range(im_alpha.shape[0]):
            for j in range(im_alpha.shape[1]):
                im_alpha[i][j][3]=50
        print("checking image")
        plt.imshow(im_alpha)


    plt.show()

def visualize_st_ellipse(image):
    csv_file_path = filedialog.askopenfilename(filetypes=[("csv File", ".csv")])
    data_tensors = pd.read_csv(csv_file_path, sep=",", index_col=False)
    X = len(data_tensors["X"].unique())
    Y = len(data_tensors["Y"].unique())

    cord_list = {
        "x_val": [],
        "y_val": [],
        "e1_vec": [],
        "e2_vec": [],
        "e3_vec": [],
        "e4_vec": [],
        "ani_val1": [],
        "ani_val2": [],
        "val1": [],
        "val2": [],
        "CL": [],
        "CP": []
    }

    for i in range(X * Y):

        if data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0:
            cord_list["x_val"].append(data_tensors["X"][i])
            cord_list["y_val"].append(data_tensors["Y"][i])
            cord_list["e1_vec"].append(data_tensors["val1"][i] * data_tensors["vec00"][i])
            cord_list["e2_vec"].append(data_tensors["val1"][i] * data_tensors["vec01"][i])
            cord_list["e3_vec"].append(data_tensors["val2"][i] * data_tensors["vec10"][i])
            cord_list["e4_vec"].append(data_tensors["val2"][i] * data_tensors["vec11"][i])
            cord_list["val1"].append((data_tensors["val1"][i]))
            cord_list["val2"].append((data_tensors["val2"][i]))
            cord_list["CL"].append((data_tensors["CL"][i]))
            cord_list["CP"].append((data_tensors["CP"][i]))

    print("Visualizing Tensor Voting")
    max_val1 = np.max(cord_list['val1'])
    min_val1 = np.min(cord_list['val1'])
    max_val2 = np.max(cord_list['val2'])
    min_val2 = np.min(cord_list['val2'])
    norm_val1 = [float(val1)/(max_val1-min_val1) for val1 in cord_list['val1']]
    norm_val2 = [float(val2)/(max_val2-min_val2) for val2 in cord_list['val2']]
    xy_cord = zip(cord_list["x_val"], cord_list["y_val"])
    ells=[]
    count =0
    c_map = plt.cm.get_cmap('coolwarm', 100)(np.array(cord_list["CL"]))
    for t in xy_cord:
        angle_val = degrees(atan(cord_list["e2_vec"][count]/cord_list["e1_vec"][count]))
        if cord_list["val1"][count]<0.1:
            temp = Ellipse(xy=t,
                            width=0.3, height=0.3,
                            angle=angle_val, color=c_map[count])
        else:
            temp = Ellipse(xy=t,
               width=norm_val1[count], height=norm_val2[count],
               angle=angle_val, color=c_map[count])

        ells.append(temp)
        count+=1
    norm = mpl.Normalize(vmin=0, vmax=1)
    fig, ax = plt.subplots()
    plt.suptitle("Structure Tensor")
    # print("image",image)
    for e in ells[::2]:
        ax.add_patch(e)

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('coolwarm', 100)))
    ax.set_xlim(0, max(cord_list["x_val"]))
    ax.set_ylim(0, max(cord_list["y_val"]))
    ax.axis('off')

    if image is not None:
        im_alpha = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        for i in range(im_alpha.shape[0]):
            for j in range(im_alpha.shape[1]):
                im_alpha[i][j][3]=70
        print("checking image")
        plt.imshow(im_alpha)


    plt.show()


def visualize_cp(image):
    csv_file_path = filedialog.askopenfilename(filetypes=[("csv File", ".csv")])
    data_tensors = pd.read_csv(csv_file_path, sep=",", index_col=False)

    X = len(data_tensors["X"].unique())
    Y = len(data_tensors["Y"].unique())
    cord_list = {
        "x_val": [],
        "y_val": [],
        "cord_val": [],
        "CL": [],
        "CP": []
    }

    a = list(map(add, data_tensors["val1"], data_tensors["val2"]))
    amin, amax = min(a), max(a)
    for i, val in enumerate(a):
        a[i] = (val - amin) / (amax - amin)

    for i in range(X * Y):
        if (data_tensors["CL"][i] != 0.0 or data_tensors["CP"][i] != 0.0) and (data_tensors["CL"][i] < 0.4) and (a[i]>0.002):
            if data_tensors['CL'][i] <= 0.05:
                cord_list["x_val"].append(data_tensors["X"][i])
                cord_list["y_val"].append(data_tensors["Y"][i])
                cord_list["cord_val"].append([data_tensors['X'][i], data_tensors["Y"][i]])
                cord_list['CL'].append([data_tensors['X'][i], data_tensors["Y"][i], 0, 0])

    if image is not None:
        im_alpha = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        for i in range(im_alpha.shape[0]):
            for j in range(im_alpha.shape[1]):
                im_alpha[i][j][3]=70
        print("checking image")
        plt.imshow(im_alpha)

    plt.scatter(cord_list['x_val'], cord_list['y_val'], s=5)
    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.show()

