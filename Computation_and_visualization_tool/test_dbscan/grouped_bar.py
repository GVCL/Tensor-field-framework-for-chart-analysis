import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("/home/komaldadhich/Desktop/test.csv", sep=',', index_col=False)
print(df.head())

months =df['Month']
data_1958 =df['1958']
data_1959 =df['1959']
data_1960 =df['1960']

x = np.arange(len(months))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, data_1958, width, label='1958', color='b')
rects2 = ax.bar(x + width/2, data_1959, width, label='1959', color='r')
rects3 = ax.bar(x + width/1.5, data_1960, width, label='60', color='k')

ax.set_ylabel('Number of passengers')
ax.set_title('Monthly transatlantic airtravel for 1958-1959')
ax.set_xticks(x)
ax.set_xticklabels(months)
# ax.scatter(months, data_1958,marker ='o', color='b', label='1958',zorder=3)
# ax.scatter(months, data_1959,marker ='o', color='r', label='1959',zorder=3)

ax.legend()
plt.show()

