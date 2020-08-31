import numpy as np
import sys
import os

from threading import Lock
from affine import config
from logging import getLogger
from affine.detection.data_processor \
    import DataProcessor, file_from_binary


logger = getLogger(__name__)

caffe_root = '/opt/caffe/'
sys.path.insert(0, caffe_root + 'python')
# This try block needs to be removed after we move everything onto a single ami
try:
    import caffe
except ImportError:
    logger.info("Caffe not found! Caffe module will not be used!")

__all__ = ['CaffeProcessor', 'DEPLOY', 'CAFFE_MODEL', 'MEAN_IMAGE',
           'TRAIN_TEST', 'SOLVER', 'LABELS']

DEPLOY, CAFFE_MODEL = ['deploy.prototxt', 'caffe_model']

MEAN_IMAGE, TRAIN_TEST, SOLVER, LABELS = ['mean.binaryproto',
                                          'train_test.prototxt',
                                          'solver.prototxt',
                                          'labels.csv']


def check_path(path_name):
    assert os.path.exists(path_name), "Non-existent path {}".format(path_name)


class CaffeProcessor(DataProcessor):

    """
        Currently this class NEEDS:
        - caffe libraries installed in /opt/caffe/
        - and the following needs to be run before using this class:
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
            PYTHONPATH=/opt/caffe/distribute/python/caffe:$PYTHONPATH
    """

    def __init__(self, root_path, caffe_model_file, deploy_model_file,
                 mean_file=None, with_gpu=False):
        """
        Args:
            root_path: directory where the remaining model files are stored
            caffe_model_file: string, .caffemodel file with pre-trained model
                for example setnet.caffemodel
            deploy_model_file: string, .prototxt file with deployment cnn configuration,
                usually deploy.prototxt
            mean_file: (optional) string (default None) .binaryproto file
                with the average image from the dataset used to train the model,
                e.g., 'coco_mean.binaryproto'.
                If we provide this param, test images will be normalized using it,
                as the training data did.
            with_gpu: (optional) boolean (default=False)

        Returns:
        Raises/Assertions:
            if caffe model or deploy file do not exist
        """
        self.lock = Lock()
        self.pretrained_caffemodel = os.path.join(root_path, caffe_model_file)
        self.caffe_deploy_modelfile = os.path.join(root_path,
                                                   deploy_model_file)
        self._layer_info = {}
        assert os.path.exists(self.pretrained_caffemodel), \
            "pretrained model file does not exist"
        assert os.path.exists(self.caffe_deploy_modelfile), \
            "deploy caffe model file does not exist"

        if mean_file is not None:
            mean_file = os.path.join(root_path, mean_file)
            self.mean_file = mean_file
            tmp_file, _ = os.path.splitext(mean_file)
            self.mean_np_file = tmp_file + '.npy'
        else:
            self.mean_file = None
            self.mean_np_file = None

        self.IMAGE_BATCH_SIZE = 50
        self.gpu_available = with_gpu
        logger.info("GPU present: {}".format(with_gpu))
        self.net = self._load_net_data()
        for layer in self.net.blobs:
            self._layer_info[layer] = self._get_descriptor_length(layer)

    def _load_net_data(self):
        """ Load necessary params on init """

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

        net = caffe.Classifier(self.caffe_deploy_modelfile,
                               self.pretrained_caffemodel,
                               mean=mean_image, gpu=self.gpu_available,
                               channel_swap=(2, 1, 0),
                               raw_scale=255,
                               image_dims=(256, 256))
        net.set_phase_test()
        return net

    def get_deploy_file(self):
        logger.info("Getting deploy file contents")
        fd = open(self.caffe_deploy_modelfile)
        deploy_contents = fd.read()
        fd.close()
        return deploy_contents

    def get_layer_info(self):
        logger.info("Getting layer info")
        return self._layer_info

    @file_from_binary
    def predict(self, list_of_images, oversample=False):
        """
        Computes prediction of current model for each image in the input list

        Args:
            list_of_imagefiles: list of image file names to extract features
            oversample: oversample param from caffe.net.predict
        Returns:
            2-D list with predictions for each input image
            Each row contains the probability of each class for one image
        Asserts:
            if list_of_images is not a list
        """
        assert (type(list_of_images) == list), \
            "Input should be a list of image files"

        num_images = len(list_of_images)
        logger.info("Running predict on {} images".format(num_images))

        step = self.IMAGE_BATCH_SIZE
        all_predictions = []
        images_left = num_images
        for start_indx in range(0, num_images, step):
            list_imgs = [caffe.io.load_image(image_file)
                         for image_file
                         in list_of_images[start_indx:start_indx + step]]

            images_in_batch = min(images_left, step)
            prediction = self._caffe_predict(list_imgs, oversample=oversample)
            prediction = np.array(prediction.flat)
            prediction = np.reshape(prediction,
                                    (images_in_batch,
                                     len(prediction) / images_in_batch))
            if len(all_predictions) == 0:
                all_predictions = prediction
            else:
                all_predictions = np.append(
                    all_predictions, prediction, axis=0)
            images_left -= step

        if type(all_predictions) == np.ndarray:
            all_predictions = all_predictions.tolist()

        return all_predictions

    @file_from_binary
    def extract(self, list_of_imagefiles, feat_layer):
        """
        Computes a descriptor for each image in the input list of imagefiles.
        The feature will be extracted from the cnn level 'feat_layer'

        Args:
            list_of_imagefiles: list of images to extract features from
            feat_layer: string, layer name from the cnn to be used
                to extract the features
        Returns:
            2-D list of float (desc_matrix) where each row is the feature
                of the corresponding image
        Raises/Assertions:
            Asserts if the feature layer is not part of current model
        """
        assert (type(list_of_imagefiles) == list), \
            "Input should be a list of image files"

        n_imgs = len(list_of_imagefiles)
        desc_length = self._layer_info[feat_layer]
        logger.info('Desc length %d . Number of images %d' %
                    (desc_length, n_imgs))
        desc_matrix = np.zeros((n_imgs, desc_length))

        for i, im_name in enumerate(list_of_imagefiles):
            desc = self._get_descriptor(im_name, feat_layer)
            desc_matrix[i, :] = desc

        return desc_matrix.tolist()

    def _get_descriptor_length(self, layer):
        """
        Get length of the descriptor obtained form the given layer

        Args:
            layer: string, cnn layer to use to extract features (e.g., 'fc8')
        Returns:
            int, length of descriptor in the input layer
        Raises:
            Asserts if the layer name is not part of the currently loaded model
        """
        assert (layer in self.net.blobs), \
            "Given layer (%s) is not part of currently loaded caffe model" % \
            (layer)

        out_shape = self.net.blobs[layer].data.shape
        # out_shape[0] is the number of input variations
        desc_length = out_shape[1] * out_shape[2] * out_shape[3]
        return desc_length

    def _get_descriptor(self, im_name, layer):
        """
        Get the descriptor for a single image

        Args:
            im_name: string, full path to input image
            layer: string, layer from the cnn to use to extract the features,
                e.g. 'fc8'
        Returns:
            1-D array with the descriptor
        Raises:
            Asserts if the layer name is not part of the currently loaded model
            or if the image file does not exist
        """
        assert (layer in self.net.blobs), \
            "Given layer (%s) is not part of currently loaded caffe model" % \
            (layer)

        assert os.path.exists(im_name), "Image does not exist"

        logger.info('computing descriptor for %s' % im_name)
        input_image = caffe.io.load_image(im_name)
        self._caffe_predict([input_image])
        feat_level = self.net.blobs[layer].data[4]
        return feat_level.flat

    @classmethod
    def load_model(cls, model_dir):
        """
        Method returns instance of class CaffeProcessor for a valid model_id

        This method downloads the model's tarball

        Args:
            model_id: primary key from table cnn_models

        Returns:
            CaffeProcessor instance loaded from s3_cnn_bucket
        """
        has_gpu = config.get('has_gpu')
        assert os.path.exists(model_dir)
        if os.path.isfile(os.path.join(model_dir, MEAN_IMAGE)):
            return cls(model_dir, CAFFE_MODEL, DEPLOY, mean_file=MEAN_IMAGE,
                       with_gpu=has_gpu)
        else:
            return cls(model_dir, CAFFE_MODEL, DEPLOY, with_gpu=has_gpu)

    def _caffe_predict(self, inputs, oversample=False):
        """
        Predict classification probabilities of inputs.

        Take
        inputs: iterable of (H x W x K) input ndarrays.
        oversample: average predictions across center, corners, and mirrors
                    when True (default). Center-only prediction when False.

        Give
        predictions: (N x C1 x C2 x C3) ndarray of class probabilities
                     for N images and C1 x C2 x C3 classes.

        THIS FUNCTION HAS BEEN COPIED FROM CAFFE IN ORDER TO FIX
        predictions = out[self.net.outputs[0]].squeeze((2,3))
        THIS SHOULD NOT BE MODIFIED FURTHER
        """
        # Scale to standardize input dimensions.
        input_ = np.zeros((len(inputs), self.net.image_dims[0],
                           self.net.image_dims[1], inputs[0].shape[2]),
                          dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = caffe.io.resize_image(in_, self.net.image_dims)

        if oversample:
            # Generate center, corner, and mirrored crops.
            input_ = caffe.io.oversample(input_, self.net.crop_dims)
        else:
            # Take center crop.
            center = np.array(self.net.image_dims) / 2.0
            crop = np.tile(center, (1, 2))[0] + np.concatenate([
                -self.net.crop_dims / 2.0,
                self.net.crop_dims / 2.0
            ])
            input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

        # Classify
        caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.net.preprocess(self.net.inputs[0], in_)
        out = self.net.forward_all(**{self.net.inputs[0]: caffe_in})
        predictions = out[self.net.outputs[0]]

        # For oversampling, average predictions across crops.
        if oversample:
            predictions = predictions.reshape((len(predictions) / 10, 10, -1))
            predictions = predictions.mean(1)

        return predictions
