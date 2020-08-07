from tkinter import *
from tkinter import Label, Tk
from visualize_tensor import *
import sys
from rebuild_bar_latest import *

root = Tk()
# w = Label(root, text= "Image uploader")
root.title("Tensor Field Visualizer")
img = None

def filebrowse():
    global img
    nwin = Toplevel()
    nwin.title("Selected Graph Image")

    path = filedialog.askopenfilename(filetypes=[("Image File", ".*")])

    img, tkimage, im = read_image(path)
    myvar = Label(nwin, image=tkimage)
    myvar.image = tkimage
    myvar.pack()


def browsefunc():
    browsebutton_csv0 = Button(
        root, text="Upload image", width=25, command=lambda:filebrowse()
    )
    print()
    browsebutton_csv0.pack()

    browsebutton_csv1 = Button(
        root, text="Visualize Structured Tensor", width=25, command=lambda: visualize_st_ellipse(img)
    )
    browsebutton_csv1.pack()

    # browsebutton_csv2 = Button(
    #     root, text="Visualize Tensor Vote post AD", width=25, command=lambda: visualize_tensor_voting_AD(img)
    # )
    # browsebutton_csv2.pack()
    #
    #
    # browsebutton_csv3 = Button(
    #     root, text="Visualize Tensor Vote before AD", width=25, command=lambda: visualize_tensor_voting(img)
    # )
    # browsebutton_csv3.pack()

    browsebutton_csv3 = Button(
        root, text="Visualize Saliency", width=25, command=lambda: visualize_colormap(img)
    )
    browsebutton_csv3.pack()

    browsebutton_csv4 = Button(
        root, text="Visualize elliptical glyphs", width=25, command=lambda: visualize_tv_ellipse(img)
    )
    browsebutton_csv4.pack()

    # browsebutton_csv5 = Button(
    #     root, text="Visualize Saliency distribution", width=25, command=lambda: visualize_distribution(img)
    # )
    # browsebutton_csv5.pack()
    browsebutton_csv8 = Button(
        root, text="Visualize critical points", width=25, command=lambda: visualize_cp(img)
    )
    browsebutton_csv8.pack()

    browsebutton_csv6 = Button(
        root, text="Reconstruct bar chart", width=25, command=lambda: reconstruct_bar(img)
    )
    browsebutton_csv6.pack()

    browsebutton_csv7 = Button(
        root, text="Reconstruct scatterplot chart", width=25, command=lambda: reconstruct_scatter(img)
    )
    browsebutton_csv7.pack()



browsefunc()

root.mainloop()
