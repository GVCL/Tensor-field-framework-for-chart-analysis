import xml.etree.ElementTree as ET
from Text_DET_REC.Retrieve_Text import get_text_labels,get_title,get_xtitle,get_ytitle
from matplotlib import pyplot as plt
import pandas as pd
import cv2


path = "/Users/daggubatisirichandana/PycharmProjects/chart_percept/Final_Reconstruct/Data/grouped_bar/gb01/"
image_name = path.split('/')
image_name = image_name[len(image_name)-2]
img = cv2.imread(path+image_name+".png")
root = ET.parse(path+image_name+'.xml').getroot()
df = pd.read_csv(path+"data.csv")
print(df.head())
label=list(df)[1:len(list(df))-1]
print(df['X'],df['bar_width'][0])

plt.plot(df[label[0]], df[label[0]], 'ko')
plt.xlabel(label[0])
plt.ylabel(label[1])
plt.title("Correlation Plot of Temperatues between Seasons")
plt.tight_layout(rect=[0, 0, 1, 0.95])

# plt.bar(df['X'], height=df[label[0]], color=[[1,0,0]])
# plt.xlabel(get_xtitle(img,root))
# plt.ylabel(get_ytitle(img,root))
# plt.title(get_title(img,root)+" in "+label[0].capitalize())
# plt.tight_layout(rect=[0, 0, 1, 0.95])

# plt.bar(df['X'], height=df[label[1]], color=[[0,0,1]])
# plt.xlabel(get_xtitle(img,root))
# plt.ylabel(get_ytitle(img,root))
# plt.title(get_title(img,root)+" in "+label[1].capitalize())
# plt.tight_layout(rect=[0, 0, 1, 0.95])

# plt.plot(df['X'], df[label[0]], 'ro')
# plt.plot(df['X'], df[label[1]], 'bo')
# plt.xlabel(get_xtitle(img,root))
# plt.ylabel(get_ytitle(img,root))
# plt.title(get_title(img,root))
# plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()
