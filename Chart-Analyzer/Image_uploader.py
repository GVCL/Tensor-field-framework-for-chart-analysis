from tkinter import *
# from Tkinter import Label, Tk
from tkinter import filedialog
from helper import *
import matplotlib
matplotlib.use("TkAgg")

root = Tk()
# w = Label(root, text= "Image uploader")
root.title("Image Uploader")
img = None

def browsefunc():
    global img
    nwin = Toplevel()
    nwin.title("Selected Graph Image")

    # path = filedialog.askopenfilename(filetypes=[("Image File", ".*")])
    path = filedialog.askopenfilename()

    img, tkimage, im = image_read(path)
    myvar = Label(nwin, image=tkimage)
    myvar.image = tkimage
    myvar.pack()

browsebutton = Button(root, text="upload image", width=25, command=browsefunc)
browsebutton.pack()
root.mainloop()
