def find_neighbour_old(i, j, X, Y, scale):
    n_list = []
    if ((j+scale) < X):
        n_list.append([i, j+scale])
    if ((j-scale) >= 0):
        n_list.append([i, j-scale])
    if ((i+scale) < Y):
        n_list.append([i+scale, j])
    if ((i-scale) >= 0):
        n_list.append([i-scale, j])

    return n_list

def find_neighbour(i, j, X, Y, scale):
    n_list = []
    for q in range(1, scale+1):
        if ((j+q) < X):
            n_list.append([i, j+q])
        if ((j-q) >= 0):
            n_list.append([i, j-q])
        if ((i+q) < Y):
            n_list.append([i+q, j])
        if ((i-q) >= 0):
            n_list.append([i-q, j])

    return n_list