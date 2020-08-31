import scipy
import cv2
import numpy as np
import math


def edge_nms(edge_map):
    """Implementation of NMS that returns mag and orientation of each pixel.

    Params: binary map of edges

    Returns:
        mag_sup: matrix of pixel magnitudes
        sobeloutdir: matrix of pixel orientations
    """
    height, width = edge_map.shape

    G = edge_map.copy()
    gradx = cv2.Sobel(G, cv2.CV_64F, 1, 0, ksize=3)
    grady = cv2.Sobel(G, cv2.CV_64F, 0, 1, ksize=3)

    sobeloutmag = scipy.hypot(gradx, grady)
    sobeloutdir = scipy.arctan2(grady, gradx)
    neg_inds = np.nonzero(sobeloutdir < 0)
    sobeloutdir[neg_inds] = math.pi + sobeloutdir[neg_inds]

    mag_sup = sobeloutmag.copy()
    for x in range(1, height-1):
        for y in range(1, width-1):
            p = sobeloutdir[x, y]
            m = sobeloutmag[x, y]
            if (p < math.pi/8 and p >= 0) or p >= 7*math.pi/8:
                if m <= sobeloutmag[x][y+1] or m <= sobeloutmag[x][y-1]:
                    mag_sup[x][y] = 0
            elif p >= 5*math.pi/8 and p < 7*math.pi/8:
                if m <= sobeloutmag[x-1][y+1] or m <= sobeloutmag[x+1][y-1]:
                    mag_sup[x][y] = 0
            elif p >= 3*math.pi/8 and p < 5*math.pi/8:
                if m <= sobeloutmag[x+1][y] or m <= sobeloutmag[x-1][y]:
                    mag_sup[x][y] = 0
            else:
                if m <= sobeloutmag[x+1][y+1] or m <= sobeloutmag[x-1][y-1]:
                    mag_sup[x][y] = 0

    return mag_sup, sobeloutdir


def box_nms(boxes, img_shape, delta=0.7):
    """Performs NMS on bounding boxes.

    Params: list of boxes and image shape
        delta: IoU threshold

    Returns: List of final boxes
    """
    final_boxes = []
    beta = delta + 0.05
    image = np.zeros(img_shape)
    for b in boxes:
        s1, s2, x, y = b
        IoU = np.sum(image[x:x+s1, y:y+s2].flatten()) / (s1*s2)
        if IoU <= 1-beta:
            final_boxes.append(b)
            image[x:x+s1, y:y+s2] = 1.
    return final_boxes
