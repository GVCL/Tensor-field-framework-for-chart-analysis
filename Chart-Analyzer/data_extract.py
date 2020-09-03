from reconstruct_chart import *

def reconstruct_chart(filename,chart_type):
    if chart_type=='Vertical_simple_bar':
        bar(filename)
    elif chart_type=='Horizontal_simple_bar':
        H_bar(filename)
    elif chart_type=='Histogram':
        hist(filename)
    elif chart_type=='Vertical_grouped_bar':
        G_bar(filename)
    elif chart_type=='Horizontal_grouped_bar':
        GH_bar(filename)
    elif chart_type=='Vertical_stacked_bar':
        S_bar(filename)
    elif chart_type=='Horizontal_stacked_bar':
        SH_bar(filename)
    print("Chart Reconstuction Done ... !")
