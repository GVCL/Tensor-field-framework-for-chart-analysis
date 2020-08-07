import numpy as np
import pandas as pd
import cv2
from itertools import groupby
# from test1 import emd_calculation
from emd_computation import emd_calc
import matplotlib.pyplot as plt
import math
from scipy.stats import wasserstein_distance

def find_neighbour(x, y, cords):
    n_list = []
    # p1 = [x+1, y]
    # p2 = [x-1, y]
    # p3 = [x-1, y-1]
    # p4 = [x-1, y+1]
    # p5 = [x+1, y-1]
    # p6 = [x+1, y+1]
    # p7 = [x, y-1]
    # p8 = [x, y+1]
    # n_list.append(p1)
    # n_list.append(p2)
    # n_list.append(p3)
    # n_list.append(p4)
    # n_list.append(p5)
    # n_list.append(p6)
    # n_list.append(p7)
    # n_list.append(p8)
    # print n_list
    p = [x, y-2]
    q = [x-1, y-1]
    r = [x+1, y-1]
    if p in cords:
        n_list.append(p)
    if q in cords:
        n_list.append(q)
    if r in cords:
        n_list.append(r)
    return n_list

'''
def find_neighbour(x, y, cords):
    n_list =[]

    p = [x, y+7]
    q = [x+13, y+1]
    r = [x+13, y+7]
    a = [x-13, y]
    b = [x-13, y-7]
    c = [x, y-6]
    d = [x, y-7]
    e = [x+13, y-6]
    f = [x+13, y]
    g = [x-13, y+6]
    h = [x-13, y-1]
    i = [x, y+6]
    if p in cords:
        n_list.append(p)
    if q in cords:
        n_list.append(q)
    if r in cords:
        n_list.append(r)
    if a in cords:
        n_list.append(a)
    if b in cords:
        n_list.append(b)
    if c in cords:
        n_list.append(c)
    if d in cords:
        n_list.append(d)
    if e in cords:
        n_list.append(e)
    if f in cords:
        n_list.append(f)
    if g in cords:
        n_list.append(g)
    if h in cords:
        n_list.append(h)
    if i in cords:
        n_list.append(i)
    return n_list

'''
# data_tensors = pd.read_csv("/home/komaldadhich/Desktop/results_poss/scatter/excel_files/tensor_vote_matrix_poss_scatter_gridless.csv", sep=",", index_col=False)
# data_tensors = pd.read_csv("/home/komaldadhich/Desktop/results_poss/scatter/excel_files/tensor_vote_matrix_poss_scatter_gridless.csv", sep=",", index_col=False)
data_tensors = pd.read_csv("tensor_vote_matrix_sc01_lab.csv", sep=",", index_col=False)

X = len(data_tensors["X"].unique())
Y = len(data_tensors["Y"].unique())
cord_list = {
        "x_val": [],
        "y_val": [],
        "cord_val": [],
        "CL": [],
        "CP": []
    }

for i in range(X*Y):
    if data_tensors['CL'][i]>0.0 and data_tensors['CL'][i]<=0.05:
        cord_list["x_val"].append(data_tensors["X"][i])
        cord_list["y_val"].append(data_tensors["Y"][i])
        cord_list["cord_val"].append([data_tensors['X'][i], data_tensors["Y"][i]])
        cord_list['CL'].append([data_tensors['X'][i], data_tensors["Y"][i], 0, 0])

cord_list1 = np.array(cord_list['cord_val'])

label = np.zeros(len(cord_list['cord_val']))
neighbour_count = np.zeros(len(cord_list['cord_val']))
cord_list_x=[]
cord_list_y=[]
count = 1
for i in cord_list1:
    x = i[0]
    y = i[1]
    n_list = find_neighbour(x , y, cord_list['cord_val'])
    n = 0
    # if label[cord_list['cord_val'].index([x, y])] == 0:
    #     label[cord_list['cord_val'].index([x, y])] = count
    #     cord_list['CL'][cord_list['cord_val'].index([x, y])][2] = count
    #     count += 1
    # for p in n_list:
    #     if p in cord_list['cord_val']:
    #         n += 1
    #         if label[cord_list['cord_val'].index(p)] == 0:
    #             label[cord_list['cord_val'].index(p)] = label[cord_list['cord_val'].index([x, y])]
    #             cord_list['CL'][cord_list['cord_val'].index(p)][2] = label[cord_list['cord_val'].index([x, y])]
    # cord_list['CL'][cord_list['cord_val'].index([x, y])][3] = n
    # neighbour_count[cord_list['cord_val'].index([x, y])] = n

    if (len(n_list) == 3):
        for p in n_list:
            x += p[0]
            y += p[1]
        cord_list_x.append(int(x / len(n_list)))
        cord_list_y.append(int(y / len(n_list)))
    # for p in n_list:
    #     x += p[0]
    #     y += p[1]


plt.scatter(cord_list['x_val'], cord_list['y_val'], s=5)
plt.show()

fig, ax = plt.subplots()
ax.scatter(cord_list_x, cord_list_y, color='k')
ax.axis('off')
plt.show()
df = pd.DataFrame(np.array(cord_list_x), columns=['X'])
df1 = pd.DataFrame(np.array(cord_list_y), columns=['Y'])
df_final = pd.concat([df, df1], axis=1)
df_final.to_csv("reconstructed_data.csv", index=False, sep=",")

# data = pd.read_csv("/home/komaldadhich/Documents/study/Research/graph_percept/source/project/chart_percept/charts/scatter/sc05/scatter_1.csv",  sep=",", index_col=False)
# data = pd.read_csv("/home/komaldadhich/Documents/study/Research/graph_percept/source/project/chart_percept/charts/scatter/sc05/scatter_1.csv",  sep=",", index_col=False)
data = pd.read_csv("/home/komaldadhich/Documents/study/Research/graph_percept/source/project/chart_percept/charts/final-charts/scatter/sc06/manaus.csv",  sep=",", index_col=False)
# data = pd.read_csv("/home/komaldadhich/Documents/study/Research/graph_percept/source/project/chart_percept/charts/final-charts/scatter/sc04/cav.csv",  sep=",", index_col=False)
# data = pd.read_csv("/home/komaldadhich/Documents/study/Research/graph_percept/source/project/chart_percept/charts/final-charts/scatter/sc03/sc03/CrohnD.csv",  sep=",", index_col=False)
# xy1 = np.array(list(map(list, zip(cord_list_x, cord_list_y))))
# xy2 = np.array(list(map(list, zip(data['height'], data['weight']))))
# print xy1.shape, xy2.shape
# emd_calculation(xy2, xy1)
# print xy1.dtype
# emd_x = emd_calc(cord_list_x, data['height'])
# emd_y = emd_calc(cord_list_y, data['y'])


# pdf_xy = np.histogram2d(cord_list_x, cord_list_y)
# data_xy = np.histogram2d(np.isfinite(data['X']), np.isfinite(data['Y']))
# emd_x1 = emd_calc(cord_list_x, data['height'])
# emd_y2 = emd_calc(cord_list_y, data['y'])
# print type(pdf_x[0]), type(data_x[0])
# emd_x = wasserstein_distance(pdf_x[0], data_x[0])
# emd_y = wasserstein_distance(pdf_y[0], data_y[0])
# pdf_xy[0].convertTo(pdf_xy[0], CV_32F);
# data_xy[0].convertTo(data_xy[0], CV_32F);

ymin = min(cord_list_y)
ymax = max(cord_list_y)
xmin = min(cord_list_x)
xmax = max(cord_list_x)

org_list_x = sorted(data['time'].tolist())
org_list_y = sorted(data['value'].tolist())

ymin_org = min(org_list_y)
ymax_org = max(org_list_y)
xmin_org = min(org_list_x)
xmax_org = max(org_list_x)

norm_y = [(float(i-ymin)/float(ymax-ymin)) for i in cord_list_y]
norm_y_org = [(float(j-ymin_org)/float(ymax_org-ymin_org)) for j in org_list_y]

norm_x = [(float(i-xmin)/float(xmax-xmin)) for i in cord_list_x]
norm_x_org = [(float(j-xmin_org)/float(xmax_org-xmin_org)) for j in org_list_x]

pdf_x = np.histogram(norm_x)
pdf_y = np.histogram(norm_y)
data_x = np.histogram(norm_x_org)
data_y = np.histogram(norm_y_org)

x_dist = wasserstein_distance(norm_x, norm_x_org)
y_dist = wasserstein_distance(norm_y, norm_y_org)

final_dis = math.sqrt(((x_dist)**2 + (y_dist)**2)/2)
print (final_dis)








