import numpy as np
import pandas as pd
import cv2
# from emd_computation import emd_calc
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy import stats



data = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Topnvispaper/data/histogram/hg-1/cav.csv",  sep=",", index_col=False)
datar = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Topnvispaper/data/histogram/hg-1/hist_data.csv",  sep=",", index_col=False)

# xy2 = np.array(list(map(list, zip(data[' "Age"'], data[' "Weight (lbs)"']))))
# xy2 = np.array(list(map(list, zip(data['X'], data['Y']))))
# emd_calculation(xy2, xy1)

org_list = sorted(data["y"].tolist())
org_list = np.histogram(org_list,bins=15)[0].tolist()#,bins=range(0,151)

# org_list = sorted(data["height"].tolist())
# org_list = np.histogram(org_list,bins=7)[0].tolist()#,bins=range(0,151)

# org_list = sorted(data["value"].tolist())
# org_list = np.histogram(org_list,bins=9)[0].tolist()#,bins=range(0,151)

# print(org_list)
# plt.hist(data["y"],bins=15,color=[(0.2,0.2,0.2)])
# plt.show()


final_cord_y = sorted(datar['Bin height (in pixels)'].tolist())

ymin = min(final_cord_y)
ymax = max(final_cord_y)

ymin_org = min(org_list)
ymax_org = max(org_list)

norm_y = [(float(i-ymin)/float(ymax-ymin)) for i in sorted(final_cord_y)]
norm_y_org = [(float(j-ymin_org)/float(ymax_org-ymin_org)) for j in org_list]

# print ("Reconstructed y:", norm_y)
# print ("Original y:", norm_y_org)

print (wasserstein_distance(norm_y, norm_y_org))


