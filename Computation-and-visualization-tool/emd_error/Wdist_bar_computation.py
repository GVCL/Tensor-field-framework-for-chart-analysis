import numpy as np
import pandas as pd
import cv2
# from emd_computation import emd_calc
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy import stats



data = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Topnvispaper/data/bars/bc01/snakes_count_10.csv",  sep=",", index_col=False)
datar = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Topnvispaper/data/bars/bc01/bar_data.csv",  sep=",", index_col=False)

# xy2 = np.array(list(map(list, zip(data[' "Age"'], data[' "Weight (lbs)"']))))
# xy2 = np.array(list(map(list, zip(data['X'], data['Y']))))
# emd_calculation(xy2, xy1)
print(data.head())
org_list = sorted(data["Game_Length"].tolist())
final_cord_y = sorted(datar['Y value (in pixels)'].tolist())

ymin = min(final_cord_y)
ymax = max(final_cord_y)

ymin_org = min(org_list)
ymax_org = max(org_list)

norm_y = [(float(i-ymin)/float(ymax-ymin)) for i in sorted(final_cord_y)]
norm_y_org = [(float(j-ymin_org)/float(ymax_org-ymin_org)) for j in org_list]

# print ("Reconstructed y:", norm_y)
# print ("Original y:", norm_y_org)

print (wasserstein_distance(norm_y, norm_y_org))


