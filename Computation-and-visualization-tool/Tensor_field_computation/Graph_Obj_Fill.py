import numpy as np
import cv2

def color_fill(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #to increase contrast so that light rgb colors also get detected
    gray = cv2.convertScaleAbs(gray, alpha=0.01, beta=0)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #sorting the contours based on area
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#cv2.findContours(thresh,1, 2)
    contours = sorted(contours, key = cv2.contourArea, reverse=False)
    bg_fill = np.zeros_like(thresh)
    for poly in contours:
        vert= np.array([list(i[0]) for i in poly], np.int32)
        mask = np.zeros_like(thresh)
        cv2.fillPoly(mask, pts = [vert], color=255)
        #to check if the contours are going to overlap on already filled contours, if so don't fill
        intersect_img = cv2.bitwise_and(bg_fill, mask, mask = None)
        #update bg_fill and fill color
        if not cv2.countNonZero(intersect_img):
            bg_fill = cv2.bitwise_or(bg_fill, mask, mask = None)
            # return color of contour edge here
            edged = cv2.Canny(mask, 30, 200)
            border_pts = np.where(edged == 255)
            borderpx_intensities=[]
            borderpx_intensities.append(img[border_pts[0], border_pts[1]])
            borderpx_intensities,borderpx_freq=np.unique(borderpx_intensities[0],return_counts=True,axis=0)

            # remove white background pixels
            borderpx_intensities=borderpx_intensities.tolist()
            borderpx_freq = list(borderpx_freq)
            if ([255,255,255] in borderpx_intensities) and (len(borderpx_intensities)>1):
                white_index = borderpx_intensities.index([255,255,255])
                borderpx_freq.pop(white_index)
                borderpx_intensities.pop(white_index)

            # find index of all occurances of most frequent color item in list
            freq_indices = np.where(borderpx_freq == max(borderpx_freq))[0]
            fill_color=[0,0,0]
            for i in freq_indices:
                fill_color = np.add(fill_color,borderpx_intensities[i])
            fill_color = fill_color/len(freq_indices)
            cv2.fillPoly(img, pts = [vert], color=tuple(fill_color))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    return thresh

