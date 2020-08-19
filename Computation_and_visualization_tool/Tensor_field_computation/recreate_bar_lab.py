import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

def find_neighbour(x, y, cords):
    n_list = []

    p = [x-1, y+1]
    q = [x+1, y-1]
    r = [x-1, y-1]
    s = [x+1, y+1]
    t = [x+2, y-2]
    u = [x-2, y+2]
    v = [x+2, y+2]
    w = [x-2, y-2]

    if p in cords:
        n_list.append(p)
    if q in cords:
        n_list.append(q)
    if r in cords:
        n_list.append(r)
    if s in cords:
        n_list.append(s)
    if t in cords:
        n_list.append(t)
    if u in cords:
        n_list.append(u)
    if v in cords:
        n_list.append(v)
    if w in cords:
        n_list.append(w)
    # if x in cords:
        n_list.append(x)
    return n_list
# data_tensors = pd.read_csv("/home/komaldadhich/Documents/study/Research/graph_percept/source/project/chart_percept/Computation-and-visualization-tool/Tensor_field_computation/tensor_vote_matrix-bar1.csv", sep=",", index_col=False)
data_tensors = pd.read_csv("tensor_vote_matrix_lab.csv")
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
    # if data_tensors['CL'][i]>0.0 and data_tensors['CL'][i]<=0.00000000000001:
    # if data_tensors['CL'][i]>0.0 and data_tensors['CL'][i]<=0.05:
        cord_list["x_val"].append(data_tensors["X"][i])
        cord_list["y_val"].append(data_tensors["Y"][i])
        cord_list["cord_val"].append([data_tensors['X'][i], data_tensors["Y"][i]])
        cord_list['CL'].append([data_tensors['X'][i], data_tensors["Y"][i], 0, 0])

cord_list1 = np.array(cord_list['cord_val'])

plt.scatter(cord_list['x_val'], cord_list['y_val'], s=5)
plt.show()

label = np.zeros(len(cord_list['cord_val']))
neighbour_count = np.zeros(len(cord_list['cord_val']))
cord_list_x=[]
cord_list_y=[]
count = 1
for i in cord_list1:
    x = i[0]
    y = i[1]
    n_list = find_neighbour(x , y, cord_list['cord_val'])
    print("list", n_list)
    if (len(n_list) == 1 or len(n_list) == 0):
        for p in n_list:
            x += p[0]
            y += p[1]
        if (len(n_list) != 0):
            cord_list_x.append(int(x/(len(n_list)+1)))
            cord_list_y.append(int(y/(len(n_list)+1)))
        else:
            cord_list_x.append(int(x))
            cord_list_y.append(int(y))

# plt.scatter(cord_list['x_val'], cord_list['y_val'], s=5)
# plt.show()
#
plt.scatter(cord_list_x, cord_list_y, s=10)
plt.show()

points = zip(cord_list_x, cord_list_y)
cords_xy = sorted(points, key = lambda x: x[0])
cords_xy1 = sorted(points, key = lambda x: x[1])

min_y = min(cord_list_y)
print ("y_min:", min_y)
diff_x = []
cord_list_x = np.unique(np.array(sorted(cord_list_x)))
final_cord_x =[]
final_cord_y =[]
cords_len = len(cords_xy1)
final_list = []
print("cords", cords_xy1)
for pts in cords_xy1:
    if pts[1] > (min_y + 5) and pts not in final_list:
        final_list.append(pts)
cords_len = len(final_list)

# y = final_list[0][1]
i=0
# for i in range(cords_len-1):
while i < cords_len-1:
    if final_list[i][1] == final_list[i+1][1]:
        temp_x = (final_list[i][0] + final_list[i+1][0])/2
        temp_y = (final_list[i][1] + final_list[i+1][1])/2
        final_cord_x.append(temp_x)
        final_cord_y.append(temp_y)
        i += 2

    else:
        i += 1
print("final", final_list, final_cord_x, final_cord_y)
print (min_y)
for i in range(len(cords_xy)-1):
    if cords_xy[i][1] >= (min_y + 5):
        # print cords_xy[i][0]
        diff = cords_xy[i+1][0] - cords_xy[i][0]
        diff_x.append(diff)

# plt.hist(diff_x)
# plt.show()
width = 30
# width = 40
fi, ax= plt.subplots()
ax.axis('on')
# for point in final_list:
#     plt.scatter(point[0], point[1], color='b', s=5)
# plt.scatter(final_cord_x, final_cord_y, s=5, color='k')
for i in range(len(final_cord_y)):
    plt.bar(final_cord_x[i], final_cord_y[i],width, color=[(0.2, 0.2, 0.2)])


plt.show()
df = pd.DataFrame(np.array(final_cord_x), columns=['X'])
df1 = pd.DataFrame(np.array(final_cord_y), columns=['Y'])
df_final = pd.concat([df, df1], axis=1)
df_final.to_csv("reconstructed_data.csv", index=False, sep=",")
xy1 = np.array(list(map(list, zip(final_cord_x, final_cord_y))))
data = pd.read_csv("/home/komaldadhich/Documents/study/Research/graph_percept/source/project/chart_percept/charts/final-charts/bars/bc04/biostats.csv",  sep=",", index_col=False)
# xy2 = np.array(list(map(list, zip(data[' "Age"'], data[' "Weight (lbs)"']))))
# xy2 = np.array(list(map(list, zip(data['X'], data['Y']))))
# emd_calculation(xy2, xy1)

print (len(cord_list_y))
ymin = min(final_cord_y)
ymax = max(final_cord_y)
print (ymax, ymin)
org_list = sorted(data[' "Weight (lbs)"'].tolist())
print (org_list)
ymin_org = min(org_list)
ymax_org = max(org_list)

norm_y = [(float(i-ymin)/float(ymax-ymin)) for i in sorted(final_cord_y)]
norm_y_org = [(float(j-ymin_org)/float(ymax_org-ymin_org)) for j in org_list]

# norm_y = np.histogram(norm_y)
# norm_y_org = np.histogram(norm_y_org)
print ("Reconstructed y:", norm_y)
print ("Original y:", norm_y_org)

# print wasserstein_distance(norm_y, norm_y_org)




