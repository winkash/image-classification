import numpy as np
import cv2
import os
from tempfile import mkdtemp


def crop_windows(images_windows):
    """Do windowed detection over given images and windows. Windows are
    extracted then warped to the input dimensions of the net.

    Params:
    images_windows: (image filename, window list) iterable.

    windows: list of (minx, miny, maxx, maxy) format

    Returns:
    window_inputs: list of {filename: image filename, window: crop coordinates,
        predictions: prediction vector} dicts.
    """
    window_inputs = []
    for image_fname, windows in images_windows:
        image = cv2.imread(image_fname)
        for window in windows:
            h = window[2] - window[0]
            w = window[3] - window[1]
            window = np.array([max(0, window[0] - h/2),
                               max(0, window[1] - w/2),
                               window[2]+h/2, window[3]+w/2])
            crop = image[window[0]:window[2], window[1]:window[3]]
            window_inputs.append(crop)
    return window_inputs


def write_to_dir(patches):
    """Create a tmp dir and write patches to img files.

    Params:
        patches: list of image patches

    Returns: list of file paths
    """
    tmp_dir = mkdtemp()
    file_list = []
    for idx, p in enumerate(patches):
        file_name = os.path.join(tmp_dir, '%s.jpg' % idx)
        cv2.imwrite(file_name, p)
        file_list.append(file_name)
    return file_list
