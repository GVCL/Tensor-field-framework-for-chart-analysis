import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib as mpl

def colorMap():
    color =[]
    RED, YELLOW, GREEN, CYAN, BLUE, MAGENTA = (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1)

    color.append(RED)
    # color.append(YELLOW)
    color.append(GREEN)
    color.append(CYAN)
    color.append(BLUE)
    color.append(MAGENTA)

    c_map = (LinearSegmentedColormap.from_list('color_code', color))

    return c_map

def colorMap_bg():
    color =[]
    RED, YELLOW, GREEN, CYAN, BLUE, MAGENTA, WHITE = (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 1, 1)

    color.append(WHITE)
    color.append(GREEN)
    color.append(BLUE)

    c_map = (LinearSegmentedColormap.from_list('color_code', color))

    return c_map

def colorMap_rg():
    color =[]
    RED, YELLOW, GREEN, CYAN, BLUE, MAGENTA, WHITE = (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 1, 1)

    color.append(BLUE)
    color.append(YELLOW)
    color.append(RED)

    c_map = (LinearSegmentedColormap.from_list('color_code', color))

    return c_map

def colorMap_b():
    color =[]
    RED, YELLOW, GREEN, CYAN, BLUE, MAGENTA, WHITE, BLACK = (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 0, 0)

    color.append(BLUE)

    c_map = (LinearSegmentedColormap.from_list('color_code', color))

    return c_map

def colorMap_r():
    color =[]
    RED, YELLOW, GREEN, CYAN, BLUE, MAGENTA, WHITE, BLACK = (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 0, 0)

    color.append(RED)

    c_map = (LinearSegmentedColormap.from_list('color_code', color))

    return c_map

def colorMap_rbb():
    color =[]
    RED, YELLOW, GREEN, CYAN, BLUE, MAGENTA, WHITE, BLACK, ORAGNGE = (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 0, 0), (0.76, 0.647, 0.811)

    color.append(BLUE)
    color.append(MAGENTA)
    color.append(RED)

    c_map = (LinearSegmentedColormap.from_list('color_code', color))

    return c_map
