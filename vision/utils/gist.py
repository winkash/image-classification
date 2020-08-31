import os
import numpy as np
import cv2
import cv2.cv as cv

from .scene_functions import average_scales, scale_filter, pad_image_symmetric
from .bovw_functions import verify_files_writable, verify_files_readable

__all__ = ['Gist', 'gist_feature_extract']

IMAGE_SIZE = 128
PIX_PAD = 32

class Gist(object):
    def __init__(self, orientations, nblocks):
        """ Creates an object type Gist
            Args:
                orientations: a list of number of orientations at each scale
                nblocks: number of blocks considered to for a grid and compute the gist
            Assertions:
                AssertionError if orientations is not a list
        """
        assert isinstance(orientations, list), "orientations must be a list"
        self.imsize = IMAGE_SIZE # square image
        self.orientations = orientations #list
        self.nblocks = nblocks
        self.padding = PIX_PAD
        self.createGabors()
        self.nfeatures = len(self.Gabors)*pow(self.nblocks, 2)

    def createGabors(self):
        """ create Gabor filters to compute gist descriptor
        """
        self.total_size = self.imsize + 2*self.padding
        self.numscales = len(self.orientations)
        self.numfilters = sum(self.orientations)
        #compute parameters
        param_filters = np.zeros((self.numfilters, 4), dtype=np.float32)
        params = 0
        for s in xrange(self.numscales):
            ori = self.orientations #total orientations
            for f in xrange(ori[s]):
                param_filters[params,:] = [0.35, 0.3/pow(1.85, s), 16.0*pow(ori[s], 2)/pow(32, 2), np.pi/ori[s]*f]
                params += 1
        #computing frequencies
        x = np.arange(-self.total_size/2, self.total_size/2)
        y = np.arange(-self.total_size/2, self.total_size/2)
        xx, yy = np.meshgrid(x, y)
        fr = np.fft.fftshift(np.sqrt(pow(xx, 2) + pow(yy, 2)))
        t = np.fft.fftshift(np.angle(xx + 1j*yy*2))
        self.Gabors = {}
        for f in xrange(self.numfilters):
            tr = t + param_filters[f,3]
            tr = tr + 2*np.pi*(tr < -np.pi).astype(float) - 2*np.pi*(tr > np.pi).astype(float)
            A = -10*param_filters[f,0]
            B = (fr/self.total_size)/(param_filters[f,1] - 1)
            C = pow(2*param_filters[f,2]*np.pi*tr, 2)
            self.Gabors[f] = np.exp(pow(A*B, 2) - C)

    def compute_gist_descriptor(self, image_path):
        """ Computes the descriptor for an image
            Args:
                image_path: path to the image
            Returns:
                g: a numpy array Gist descriptor
        """
        pix = scale_filter(image_path, self.imsize)
        pix = np.float32(pix)
        N = 1 # image is grayscale
        self.pix = pix
        wblock = pow(self.nblocks, 2)
        g = np.zeros([wblock*self.numfilters, N])
        step = 0
        bg = pad_image_symmetric(self.padding, pix)
        bg = np.float32(np.real(np.fft.fft2(bg)))

        for gb in self.Gabors:
            ig = bg*np.tile(self.Gabors[gb], [N,1,1])
            ig = np.absolute(np.fft.ifft2(ig))
            ig = ig[:, self.padding:bg.shape[0] - self.padding, self.padding:bg.shape[1] - self.padding]
            avg =  average_scales(ig, self.nblocks)
            g[step:step + wblock, :] = avg.reshape((wblock, N))
            step += wblock
        return g.transpose()


def gist_feature_extract(in_file, out_file, orientations, blocks):
    """ Extract gist descriptors for every image in in_file
        Args:
            in_file: infofile
            out_file: xml with gist descriptors (opencv format)
            orientations: a list of orientations per scale [8, 8, 8, 8]
            nblocks: number of blocks to partition the image (grid)
    """
    verify_files_readable(in_file)
    verify_files_writable(out_file)
    with open(in_file, 'r') as fo:
        files = fo.read().splitlines()
    g = Gist(orientations, blocks)
    features = np.zeros((len(files), g.nfeatures), dtype=np.float32)

    for idx, f in enumerate(files):
        img = f.split()
        descriptor = g.compute_gist_descriptor(img[0])
        features[idx,:] = descriptor
    final = cv.fromarray(features)
    cv2.cv.Save(out_file, final)
