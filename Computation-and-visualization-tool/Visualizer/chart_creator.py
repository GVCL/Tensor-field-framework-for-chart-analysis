import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("/home/komaldadhich/Documents/study/Research/graph_percept/source/project/chart_percept/charts/final-charts/histogram/hg-3/manaus.csv")
# df = pd.read_csv("/home/komaldadhich/Documents/study/Research/graph_percept/source/project/chart_percept/Computation-and-visualization-tool/Tensor_field_computation/reconstructed_data.csv")
#index = df['Index']
x=df['time']
y=df['value']
data =  df[['time','value']]
# print data
points = zip(x.tolist(), y.tolist())
freq_list = sorted(points, key = lambda x: x[0])
# print freq_list
d = {}
for i in range(len(freq_list)-1):
    if freq_list[i][1] not in d:
        d[freq_list[i][1]] = (freq_list[i][0], freq_list[i+1][0])
data1=[]
for i, j in y.value_counts().items():
    print(i, j)
    data1.append([i,j])

print(data1)
df1 = pd.DataFrame(data1, columns =['height', 'freq'])
# print (y.value_counts())
print(df1)

df1.to_csv("frequency.csv", index=False, sep=",")

# print type(y.value_counts())
'''
df_freq = pd.value_counts(df['y']).to_frame().reset_index()
print (df_freq)
'''
# y2=df['Y2']
# plt.plot(df['Index'],df['X'],c='blue')
# plt.plot(df['Index'],df['Y'],c='red')
fig,a = plt.subplots()
# a.scatter(x, y,c='black', marker = 'o', zorder=3)
# plt.grid(zorder=0)
# plt.show()
#fig.patch.set_visible(False)
# a.axis('off')
'''
a.xaxis.set_visible(False)
a.yaxis.set_visible(False)
'''
# for i in range(50, 55):
# width1=30#, 46
# width2=70# 7
width=15
# plt.bar(df1['height'], df1['freq'], width, color=[(0.2, 0.2, 0.2)], zorder=3)
# plt.grid(zorder=0)
# plt.show()

plt.hist(y, bins=9, color = [(0.2, 0.2, 0.2)])#, edgecolor ='k')
# plt.xlabel("y values")
# plt.axis("off")
# width = 2
#
# fig, a = plt.subplots()
# width = 0.25
'''
p1 = a.bar(index - width/2, x, width, label='X', color = '#696969', edgecolor='k')
p2 = a.bar(index + width/2, y, width, label='Y', color='#808080', edgecolor='k')
#ax.legend((p1[0], p2[0]), ('X', 'Y'))
'''

# p1 = a.bar(y+width, y, width, color = '#2F4F4F')
#
# p1 = a.bar(y+width, y, width, color = [(0.2, 0.2, 0.2)])
#ax.autoscale_view()
plt.title("Average Heights of the Rio Negro river at Manaus")
plt.show()
