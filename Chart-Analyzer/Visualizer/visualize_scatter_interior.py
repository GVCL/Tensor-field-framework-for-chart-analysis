import matplotlib.pyplot as plt
import matplotlib

x = [2,4,6,8,10]
y = [0]*len(x)


y1 =[i+300 for i in y]
y2 =[i+600 for i in y]
y3 =[i+900 for i in y]

s = [matplotlib.rcParams['lines.markersize']**n for n in range(len(x))]
fig, ax = plt.subplots()
ax.axis('off')
plt.scatter(x,y1,s=s, marker ='o', color='k')
plt.scatter(x,y2,s=s, marker =',', color='k')
plt.scatter(x,y3,s=s, marker ='^', color='k')
plt.ylim(100, 1100)
plt.show()