import os
import numpy as np

from sklearn import cross_validation
from logging import getLogger

from affine.model import Video
from affine import config
from ...utils import scene_functions

__all__ = ['SceneDataset', 'ImageSet']

logger = getLogger(__name__)

class SceneDataset(object):
    def __init__(self, folder, labels):
        assert isinstance(labels, dict), "Labels should be a dictionary of the form {pos: 1, neg:-1}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.dataset = {}
        self.infofile = {}
        for l in labels:
            self.dataset[l] = ImageSet(l)
        self.data_folder = folder
        self.labels = labels
        self.infofile['train'] = None
        self.infofile['test'] = None

    def create_dataset(self, pos_dataset, neg_dataset, split_ratio, neg_pos_ratio):
        """ Create dataset from files
            Args:
                pos_dataset: path to file with positive frames
                neg_dataset: path to file with neg frames
                split_ratio: # test / (# train + # test)
                neg_pos_ratio: # neg / # pos
            Assertions:
                AssertionErrors: if files don't exist
        """
        assert isinstance(split_ratio, float), "Wrong split ratio"
        logger.info("creating positive dataset")
        self.create_set_from_file(pos_dataset, 'pos')
        num_neg = int(round(neg_pos_ratio * self.dataset['pos'].num_images))
        logger.info("creating negative dataset")
        self.create_set_from_file(neg_dataset, 'neg', limit=num_neg)
        self.create_partitions(split_ratio)
        self.create_infofiles('train')
        logger.info("creating training/testing partitions")
        self.create_infofiles('test')

    def create_set_from_file(self, image_file, label, limit=None):
        assert isinstance(label, str), "Label should be a string"
        assert label in self.labels.keys(), "Label is not part of the dataset"
        with open(image_file, 'r') as fo:
            lines = fo.read().splitlines()
        self.dataset[label].folder =  os.path.join(self.data_folder, label)
        if not os.path.exists(self.dataset[label].folder):
            os.makedirs(self.dataset[label].folder)
        data = []
        for line in lines:
            if limit is not None and len(data) >= limit:
                break
            v, t = line.split('\t')
            assert v, "info file formating problem "
            assert len(v) == 12, "info file formating problem "
            assert isinstance(int(v), int), "info file formating problem "
            assert t, "info file formating problem "
            assert len(t) == 12, "info file formating problem "
            assert isinstance(int(t), int), "info file formating problem "
            v = int(v)
            t = int(t)
            localpath = os.path.join(self.dataset[label].folder, "%012d_%012d.jpg" % (v,t))
            vid = Video.get(v)
            vid.download_image(t, localpath)
            data.append([v, t])
            if label == 'pos':
                if limit is not None and len(data) >= limit:
                    break
                localpath_flipped = os.path.join(self.dataset[label].folder, "%012d_%012d.jpg" % (-1*v, t))
                scene_functions.flip_image(localpath, localpath_flipped)
                data.append([-1*v, t])
        self.dataset[label].num_images = len(data)
        data = np.array(data)
        self.dataset[label].indices = data

    def create_partitions(self, ratio):
        assert ratio >= 0, 'Percentage of images in test set should be positive'
        assert isinstance(ratio, float), 'Percentage of images in test set should be positive float'
        for l in self.labels.keys():
            assert self.dataset[l].num_images > 0, "Not enough images to create partitions"
            posset = self.dataset[l]
            rs = np.random.random_integers(100)
            if ratio > 0.5:
                ratio = 1 - ratio
            train, test = cross_validation.train_test_split(posset.indices, test_size=ratio, random_state=rs)
            self.dataset[l].dsets['train'] = train
            self.dataset[l].num_train = train.shape[0]
            self.dataset[l].dsets['test'] = test
            self.dataset[l].num_test = test.shape[0]

    def create_infofiles(self, ds):
        assert ds in self.infofile.keys(), "Supporting only train and test infofiles, %s not supported" % ds
        infofile =  os.path.join(self.data_folder, ds + '_infofile.txt')
        with open(infofile, 'w') as fo:
            for l in self.labels:
                data = self.dataset[l].dsets[ds]
                assert data.shape[0] > 0, "Not enough images to create infofiles"
                for i in xrange(data.shape[0]):
                    img = "%012d_%012d.jpg" % (data[i,0], data[i,1])
                    path = os.path.join(self.dataset[l].folder, img)
                    path = os.path.abspath(path)
                    line = '%s %i %012d %012d\n' % (path, self.labels[l], data[i,0], data[i,1])
                    fo.write(line)
        self.infofile[ds] = infofile

class ImageSet(object):
    def __init__(self, label):
        assert isinstance(label, str), "Label should be a string"
        self.label = label
        self.num_images = 0
        self.folder = None
        self.indices = None
        self.dsets = {'train': np.array([]), 'test': np.array([])}
        self.num_train = 0
        self.num_test = 0
