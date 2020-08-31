import os
import cv2
import datetime
import tempfile
import shutil
import numpy as np
import pickle

from logging import getLogger

from affine import config
from affine.model import *
from affine.aws.s3client import *
from script.image_annotation_tier1 import *
from .clustering import SceneClustering
from affine.detection.utils.run_url_injection import \
    query_video_urls, check_videos_downloaded
from ...utils import bovw_functions
from ...utils.gist import gist_feature_extract
from ...utils.scene_functions import sample_frames, get_config, \
        get_params_from_config, SCENE_CFG_SPEC, \
        compute_spatial_features_from_xml


__all__ = ['SceneDiscovery']

logger = getLogger("affine.scene_discovery")

#TODO: This has to move to a config
EVERY_N_SEC = 10 #sampling video
IMGS = 20

class SceneDiscovery(object):
    def __init__(self, config_file, folder):
        """ Args:
                folder: where the images will be downloaded
                config_file: config file where all params for features are set
        """
        self.config_file = config_file
        assert os.path.exists(config_file), "File doesn't exist"
        self.scene_params = get_config(self.config_file, SCENE_CFG_SPEC.split('\n'))
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            logger.info("creating folder %s" % self.folder)
        self.max_vids = int(self.scene_params["url_injection"]["max_vids"])
        self.image_set = None
        self.urls = None
        self.params = None
        self.infofile = None
        self.final_dir = None
        self.clustering = None

    @classmethod
    def load_from_file(cls, filename):
        """ Args:
                filename: string
            Returns:
                scene: unpickled SceneDiscovery object
        """
        with open(filename, 'rb') as fo:
            scene = pickle.load(fo)
            return scene

    @classmethod
    def save_to_file(cls, scene_discovery, filename):
        """ Args:
                scene_discovery: SceneDiscovery instance to be pickled
                filename: string, name of the pickle file
            Assertions:
                AssertionError if object is not instance of SceneDiscovery
        """
        assert isinstance(scene_discovery, SceneDiscovery), "Object should be an instance of SceneDiscovery"
        with open(filename, 'wb') as fo:
            pickle.dump(scene_discovery, fo)

    def ingest_videos_youtube(self):
        """ Get videos from youtube using queries in config_file
        """
        timestamp = datetime.now()
        n_urls = query_video_urls(self.scene_params, self.folder)
        logger.info(" %s videos in the queue" % n_urls)
        logger.info("checking if videos have been downloaded")
        self.urls_file = os.path.join(self.folder, 'all_urls.txt')
        n_videos = check_videos_downloaded(self.urls_file, self.scene_params, timestamp)
        logger.info("Downloaded %d videos" % n_videos)
        with open(self.urls_file, 'r') as f:
            self.urls = f.read().splitlines()

    def get_videos_from_inventory(self, label_id):
        """ Get videos in our inventory that MTurk or WebPageLabelResult tags with label_id
            Args:
                label_id: Label id of the category to obtain videos from
        """
        video_ids = []
        d = VideoCollageEvaluator.query.filter_by(target_label_id = label_id).first()
        if d:
            query = session.query(MTurkVideoResult.video_id).filter_by(
                    evaluator_id = d.id, result = True).limit(self.max_vids)
            video_ids.extend([v for (v,) in query])
        query = session.query(VideoOnPage.video_id).join(
                WebPageLabelResult, (WebPageLabelResult.page_id == VideoOnPage.page_id)).filter(
                WebPageLabelResult.label_id == label_id).limit(self.max_vids-len(video_ids))
        video_ids.extend([v for (v,) in query])
        self.image_set = sample_frames(video_ids, IMGS, EVERY_N_SEC)
        logger.info("Videos obtained from inventory")

    def get_video_from_urls(self):
        """ Get videos from a list of urls
            Assertions:
                AssertionError if the instance doesn't have a list of urls
        """
        assert self.urls, "Need to ingest videos first, no urls found"
        query = session.query(Video.id.distinct()).join(VideoOnPage).filter_by(active=True, is_preroll=False)
        query = query.join(WebPage).filter(WebPage.remote_id.in_(self.urls))
        video_ids = [v for (v,) in query]
        self.image_set = sample_frames(video_ids, IMGS, EVERY_N_SEC)
        logger.info("Videos obtained from urls")

    def download_data(self, folder):
        """ Download frames locally
            Args:
                folder: where to download the frames
            Assertions:
                AssertionError if the instance doesn't have a list of (video, timestamp)
        """
        assert self.image_set, "Need to get videos first, no image set found"
        self.image_dir = folder
        for v, t in self.image_set:
            localpath = os.path.join(self.image_dir, "%012d_%012d.jpg" % (v,t))
            Video.get(v).download_image(t, localpath)
            logger.info(localpath)

    def compute_descriptors(self, folder):
        """ Computes a descriptor for each image in folder
            Args:
                folder: where the features will be stored
            Assertions:
                ValueError if the config file sets an unknown descriptor
        """
        self.params = get_params_from_config(self.scene_params)
        type_desc = self.params['in_feature_type']

        self.final_dir = folder
        self.infofile = os.path.join(self.folder,'info.txt')
        filenames = []
        fo = open(self.infofile, 'w')
        for filename in os.listdir(self.image_dir):
            fi, ext = os.path.splitext(filename)
            vid, tim = fi.split('_')
            line_item = '%s %s %s %s' % (os.path.join(self.image_dir, filename), '1', vid , tim)
            filenames.append((vid, tim))
            fo.write(line_item + '\n')
        fo.close()
        logger.info("info file written at %s " % self.infofile)
        self.image_set = filenames
        logger.info("computing %s descriptors" % type_desc)
        if type_desc == 'SURF' or type_desc == 'SURFEX':
            self.features = self.compute_descriptors_BOW()
        elif type_desc == 'GIST':
            self.features = self.compute_descriptors_gist()
        else:
            raise ValueError("unknown descriptor %s" % type_desc)

    def compute_descriptors_BOW(self):
        """ Computes descriptors using SURF and the BOW framework
        """
        bin_dir = config.bin_dir()
        temp_dir = tempfile.mkdtemp()
        extract_filename = os.path.join(temp_dir, 'extract.xml')
        pca_filename = os.path.join(self.final_dir, 'pca.xml')
        projected_filename = os.path.join(temp_dir, 'projected.xml')
        vocab_filename = os.path.join(self.final_dir, 'vocab.xml')
        #consider removing unnecessary files and variables
        logger.info("extracting features")
        bovw_functions.feature_extract(bin_dir, self.infofile, extract_filename, self.params)
        logger.info("computing PCA")
        bovw_functions.pca_computation(bin_dir, extract_filename, self.params['pca_dimensions'], pca_filename)
        logger.info("projecting features")
        bovw_functions.pca_projection(bin_dir, extract_filename, pca_filename, projected_filename)
        logger.info("computing vocabulary")
        bovw_functions.vocab_kms(bin_dir, projected_filename, self.params['vocab_size'], vocab_filename)
        logger.info("computing spatial scene (SURF) descriptors")
        num_levels = self.scene_params['train_detector_params']['num_levels']
        return compute_spatial_features_from_xml(extract_filename,
                                                 projected_filename,
                                                 vocab_filename,
                                                 num_levels)

    def compute_descriptors_gist(self):
        """ Computes gist descriptors for a set of images
        """
        feature_file = os.path.join(self.final_dir, 'gist.xml')
        logger.info("computing scene (GIST) descriptors")
        gist_feature_extract(self.infofile, feature_file, self.params['orientations'], self.params['nblocks'])
        features = np.asarray(cv2.cv.Load(feature_file))
        return features[:, 1:-2]

    def cluster_scenes(self, distance, method):
        """ Cluster images using a specific distance and clustering method
            Args:
                distance: string, 'mahalanobis', 'euclidean', 'chisqr', 'intersection'
                method: string, 'ward', 'complete', 'single', 'median', 'centroid', 'average'
        """
        logger.info("performing clustering")
        clu = SceneClustering(distance, method)
        clu.compute_distance(self.features)
        clu.hierarchical_clustering()
        clu.select_clusters()
        self.clustering = clu
        self.data_dim = self.features.shape

    def save_img_cluster(self, folder, indice_hash):
        """ Save images locally of a given cluster
            Args:
                folder: string, where the images will be saved
                indice_hash: the string that identifies a particular cluster
        """
        tmp_dir = os.path.join(folder, indice_hash)
        os.mkdir(tmp_dir)
        indices = self.clustering.all_clusters[indice_hash].indices
        for idx in indices:
            img = os.path.join(self.folder, '%s_%s.jpg' % (self.image_set[idx][0], self.image_set[idx][1]))
            shutil.copy(img, tmp_dir)

    def save_file_cluster(self, folder, indice_hash):
        """ Save a file containing videoid timestamp  locally of a given cluster
            Args:
                folder: string, where the file will be saved
                indice_hash: the string that identifies a particular cluster
        """
        if not os.path.exists(folder):
            os.mkdir(folder)
        indices = self.clustering.all_clusters[indice_hash].indices
        with open(os.path.join(folder, indice_hash + '.txt'), 'w') as fo:
            for idx in indices:
                line = '%s\t%s\n' % (self.image_set[idx][0], self.image_set[idx][1])
                fo.write(line)
