from Computation_and_visualization_tool.Data_Extraction.reconstruct_bar import bar
from Computation_and_visualization_tool.Data_Extraction.reconstruct_Hbar import H_bar
from Computation_and_visualization_tool.Data_Extraction.reconstruct_hist import hist
from Computation_and_visualization_tool.Data_Extraction.reconstruct_stacked_bar import S_bar
from Computation_and_visualization_tool.Data_Extraction.reconstruct_Hstacked_bar import SH_bar
from Computation_and_visualization_tool.Data_Extraction.reconstruct_grouped_bar import G_bar
from Computation_and_visualization_tool.Data_Extraction.reconstruct_Hgrouped_bar import GH_bar


def reconstruct_chart(filename,chart_type):
    if chart_type=='bar':
        bar(filename)
    elif chart_type=='H_bar':
        H_bar(filename)
    elif chart_type=='hist':
        hist(filename)
    elif chart_type=='G_bar':
        G_bar(filename)
    elif chart_type=='GH_bar':
        GH_bar(filename)
    elif chart_type=='S_bar':
        S_bar(filename)
    elif chart_type=='SH_bar':
        SH_bar(filename)
    print("Chart Reconstuction Done ... !")
