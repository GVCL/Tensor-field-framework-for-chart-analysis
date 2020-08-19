import numpy as np
import pandas as pd
import math
import cv2
# from emd_computation import emd_calc
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy import stats



data = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Topnvispaper/data/scatter/sc06/manaus.csv",  sep=",", index_col=False)
datar = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Topnvispaper/data/scatter/sc06/scatter_data.csv",  sep=",", index_col=False)

# xy2 = np.array(list(map(list, zip(data[' "Age"'], data[' "Weight (lbs)"']))))
# xy2 = np.array(list(map(list, zip(data['X'], data['Y']))))
# emd_calculation(xy2, xy1)


cord_list_y = sorted(datar['Y value (in pixels)'].tolist())
cord_list_x = sorted(datar['X value (in pixels)'].tolist())

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
print(final_dis)

