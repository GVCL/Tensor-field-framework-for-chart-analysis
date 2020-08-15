import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import colorcet as cc

df = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Generated_data/hgb/Sheet 2-Table 1.csv")
xlabels = (df.loc[ : , list(df)[0]]).values
ylabels = list(df)[1:len(list(df))-3]
data = (df.loc[ : , ylabels]).values
x = np.arange(len(xlabels))  # the label locations
# color_id = np.arange(len(ylabels))/(len(ylabels)-1)
# cmap = matplotlib.cm.get_cmap('Dark2')
# colors=cc.CET_I1
# colors=cc.CET_L15
l = (len(ylabels)-1)
graycolors=[[(191*i/l),191*i/l,191*i/l] for i in range(len(ylabels))]
graycolors=(np.array(graycolors)+32)/255
print(graycolors)


# # width = 0.2 # the width of the bars
# width = 0.25 # 2
# # width = 0.2 # 3
# # width = 0.15 #4
# fig, ax = plt.subplots(figsize=(8, 6.4))
# # fig, ax = plt.subplots(figsize=(10.4, 8))#4
# for i in range(len(ylabels)):
#     k = ((1-len(ylabels))/2+i)*width
#     ax.bar(x+k, height=data[:,i], width=width, label=ylabels[i], edgecolor='k', color=graycolors[i], linewidth = 2)
#     # ax.bar(x+k, height=data[:,i], width=width, label=ylabels[i], color=cmap(color_id[i]))
# # Add some text for labels, title and custom x-axis tick labels, etc.
# plt.xlabel(df['X-title'][0])
# plt.ylabel(df['Y-title'][0])
# plt.title(df['Title'][0])
# plt.xticks(x, xlabels, fontsize=10)
# # plt.xticks(x, xlabels, fontsize=7.5)#4
# plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
# fig.tight_layout()
# plt.savefig('/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Generated_data/grouped_bar/gb02/gb02.png')


# width = 0.4 # the width of the bars
width = 0.17 # 2
# width = 0.25 # 3
# width = 0.15 # 4
fig, ax = plt.subplots(figsize=(6.4,8))
for i in range(len(ylabels)):
    k = ((len(ylabels)-1)/2-i)*width
    ax.barh(x+k, width=data[:,i], height=width, label=ylabels[i], edgecolor='k', color=graycolors[i], linewidth = 2)
# Add some text for labels, title and custom x-axis tick labels, etc.
plt.xlabel(df['X-title'][0])
plt.ylabel(df['Y-title'][0])
# plt.title(df['Title'][0])
plt.title(df['Title'][0], fontsize=10)#2
plt.yticks(x, xlabels, fontsize=10)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
fig.tight_layout()
plt.savefig('/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Generated_data/H_grouped_bar/h_gb02/h_gb02.png')


# width = 0.7 # the width of the bars
# fig, ax = plt.subplots(figsize=(8, 6.4))
# # plt.bar(x, data[:,0], width=width, label=ylabels[0])
# plt.bar(x, data[:,0], width=width, label=ylabels[0], edgecolor='k', color=graycolors[0],linewidth = 2)
# for i in range(1,len(ylabels)):
#     # plt.bar(x, data[:,i], bottom=np.sum(data[:,:i],axis=1), width=width, label=ylabels[i])
#     plt.bar(x, data[:,i], bottom=np.sum(data[:,:i],axis=1), width=width, label=ylabels[i], edgecolor='k', color=graycolors[i],linewidth = 2)
# # Add some text for labels, title and custom x-axis tick labels, etc.
# plt.ylabel(df['X-title'][0])
# plt.xlabel(df['Y-title'][0])
# plt.title(df['Title'][0])
#
# # plt.xticks(x, xlabels, fontsize=10)
# # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., prop={"size":10})
#
# plt.xticks(x, xlabels, fontsize=7)#2
# plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., prop={"size":7})
#
# fig.tight_layout()
# plt.savefig('/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Generated_data/stacked_bar/sb02/sb02.png')


# width = 0.7 # the width of the bars
# fig, ax = plt.subplots(figsize=(8, 6.4))
# plt.barh(x, width=data[:,0], height=width, label=ylabels[0], edgecolor='k', color=graycolors[0], linewidth = 2)
# for i in range(1,len(ylabels)):
#     plt.barh(x, width=data[:,i], left=np.sum(data[:,:i],axis=1), height=width, label=ylabels[i], edgecolor='k', color=graycolors[i], linewidth = 2)
# # Add some text for labels, title and custom x-axis tick labels, etc.
# plt.xlabel(df['X-title'][0])
# plt.ylabel(df['Y-title'][0])
# plt.title(df['Title'][0])
# plt.yticks(x, xlabels, fontsize=10)
# plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., prop={"size":10})
# fig.tight_layout()
# plt.savefig('/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Generated_data/H_stacked_bar/h_sb03/h_sb03.png')


# df = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Generated_data/h/Sheet 3-Table 1.csv")
# width = 0.
# head=list(df)
# fig, ax = plt.subplots(figsize=(6.4, 4.8))
# plt.barh(df[head[0]],width=df[head[1]],height=width, color=[[0.2,0.2,0.2]])
# plt.suptitle(df['Title'][0])
# plt.xlabel(head[1])
# plt.ylabel(head[0])
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig('/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Generated_data/H_bar/h_bc03/h_bc03.png')








# df = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Generated_data/bar/bc02/orig_data02.csv")
# width = 0.35 #2
# # width = 0.5
# head=list(df)
# plt.bar(df[head[0]],height=df[head[1]],width = width, color=[[0.2,0.2,0.2]])
# plt.suptitle(df['Title'][0])
# plt.xlabel(head[0])
# plt.ylabel(head[1])
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig('/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Generated_data/bar/bc02/bc02.png')

# df = pd.read_csv("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Generated_data/hist/hg-1/orig_data-1.csv")
# head=list(df)
# plt.hist(df[head[0]],bins=15,color=[(0.2,0.2,0.2)])
# # plt.hist(df[head[0]],bins=7,color=[(0.2,0.2,0.2)])
# # plt.hist(df[head[0]],bins=9,color=[(0.2,0.2,0.2)])
# plt.suptitle(df['Title'][0])
# plt.xlabel(head[0])
# plt.ylabel("Frequency")
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig('/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Generated_data/hist/hg-1/hg-1.png')

plt.show()
