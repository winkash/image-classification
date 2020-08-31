import cv2
import os
from tempfile import mkdtemp

from affine import config
from affine.video_processing import run_cmd


def structured_edge_detection(im_file):
    bin_dir = os.path.join(config.bin_dir(), 'structured_edge_detection')
    output = os.path.join(mkdtemp(), 'output.jpg')
    run_cmd([os.path.join(bin_dir, 'edges'), '-i', im_file, '-o',
             output, '-m', os.path.join(bin_dir, 'model.yml')])
    edges = cv2.imread(output, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    return edges
