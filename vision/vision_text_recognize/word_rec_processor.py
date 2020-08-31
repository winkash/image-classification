import numpy as np
import scipy.io
import os
import sys

from affine import config
from logging import getLogger
from affine.detection.cnn.caffe_processor import CaffeProcessor, MEAN_IMAGE

logger = getLogger(__name__)

caffe_root = '/opt/caffe/'
sys.path.insert(0, caffe_root + 'python')
# This try block needs to be removed after we move everything onto a single ami
try:
    import caffe
except ImportError:
    logger.info("Caffe not found! Caffe module will not be used!")

__all__ = ['WordRecProcessor', 'DEPLOY', 'MAT_FILE']

DEPLOY, MAT_FILE = ['deploy.prototxt', 'mat_file.mat']


class WordRecProcessor(CaffeProcessor):
    """
    Inherits from CaffeProcessor class to handle loading the Word Rec CNN
    model from a .mat file.  Instead of a caffemodel file, takes in a .mat file
    """
    @classmethod
    def load_model(cls, model_dir):
        """
        Method returns instance of class WordRecProcessor

        Args:
            model_dir: dir string with model

        Returns:
            WordRecProcessor instance
        """
        has_gpu = config.get('has_gpu')
        assert os.path.exists(model_dir)
        if os.path.isfile(os.path.join(model_dir, MEAN_IMAGE)):
            return cls(model_dir, MAT_FILE, DEPLOY, mean_file=MEAN_IMAGE,
                       with_gpu=has_gpu)
        else:
            return cls(model_dir, MAT_FILE, DEPLOY, with_gpu=has_gpu)

    def _load_net_data(self):
        """
        Grab corresponding cnn layers from .mat file and assign them manually
        to an empty model.
        """
        if self.mean_file and self.mean_np_file:
            if not os.path.exists(self.mean_np_file):
                blob = caffe.proto.caffe_pb2.BlobProto()
                data = open(self.mean_file, 'rb').read()
                blob.ParseFromString(data)
                arr = np.array(caffe.io.blobproto_to_array(blob))
                np.save(self.mean_np_file, arr[0])
            mean_image = np.load(self.mean_np_file)
        else:
            mean_image = None

        mat = scipy.io.loadmat(self.pretrained_caffemodel)
        layers = mat['layers']

        # load params into model
        net = caffe.Net(self.caffe_deploy_modelfile)

        conv_layers = {
            'conv1': layers[0, 0][0, 0],
            'conv2': layers[0, 3][0, 0],
            'conv3': layers[0, 6][0, 0],
            'conv3.5': layers[0, 8][0, 0],
            'conv4': layers[0, 11][0, 0]
        }
        fc_layers = {
            'fc5': layers[0, 13][0, 0],
            'fc6': layers[0, 15][0, 0],
            'fc-class': layers[0, 17][0, 0]
        }
        for cl, ml in conv_layers.iteritems():
            net.params[cl][0].data[...] = ml[4].transpose(
                [1, 0, 2, 3]).transpose()
            n = ml[3].shape[1]
            net.params[cl][1].data[...] = ml[3].reshape(1, 1, 1, n)

        for cl, ml in fc_layers.iteritems():
            a, b, c, d = ml[4].shape
            n = ml[3].shape[1]
            net.params[cl][0].data[...] = ml[4].transpose(
                [1, 0, 2, 3]).reshape(
                a*b*c, d, order='F').transpose().reshape(1, 1, d, a*b*c)
            net.params[cl][1].data[...] = ml[3].reshape(1, 1, 1, n)

        net.set_phase_test()
        net.set_channel_swap(net.inputs[0], (2, 1, 0))
        if self.gpu_available:
            net.set_mode_gpu()
        else:
            net.set_mode_cpu()
        net.set_raw_scale(net.inputs[0], 255)
        if mean_image:
            net.set_mean(net.inputs[0], mean_image)
        net.image_dims = (32, 100)
        return net
