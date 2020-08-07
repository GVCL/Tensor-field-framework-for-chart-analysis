import numpy as np
import matplotlib.pylab as pl
# import ot
# import ot.plot
import cv2

def emd_calculation(xs, xt):
    # print xs.shape, xt.shape
    n = len(xt)  # nb samples

    a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

    # loss matrix
    M = ot.dist(xs, xt)
    M /= M.max()

    pl.figure(2)
    pl.imshow(M, interpolation='nearest', cmap= 'gray')
    pl.title('Cost matrix M')
    pl.show()

    G0 = ot.emd(a, b, M)
    print( G0.shape)

    pl.figure(3)
    pl.imshow(G0[::-1], interpolation='nearest', cmap = 'gray')
    pl.title('OT matrix G0')
    pl.show()

    pl.plot(G0)
    pl.show()

