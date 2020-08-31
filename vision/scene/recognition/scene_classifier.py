import os
import tempfile
import datetime
import shutil

from configobj import ConfigObj
from logging import getLogger

from affine import config
from affine.aws.s3client import upload_to_s3
from affine.model import session, Label, LabelHash, WebPageLabelResult, \
         WebPage, VideoOnPage, SceneDetector, MTurkImage, TrainingImage
from affine.detection.utils import AbstractInjector
from affine.detection.utils.get_training_data import get_negative_examples
from .scene_dataset import SceneDataset
from ...utils import scene_functions, bovw_functions

LABELS = {'pos': 1, 'neg': -1}
IMGS = 1
EVERY_N_SEC = 10

logger = getLogger(__name__)

class SceneClassifier(object):
    def __init__(self, config_file, folder):
        """
            Args:
            config_file: string, path to config file to train a scene classifier
            folder: string, path to folder where all the data will be found
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.final_dir = folder
        assert os.path.exists(config_file), "Config file doesn't exist"
        self.config_file =  config_file
        self.feat_params = scene_functions.get_params_from_config(ConfigObj(config_file))
        self.bin_dir = config.bin_dir()

    def set_params(self):
        self.scene_params = scene_functions.get_config(self.config_file, scene_functions.SCENE_CFG_SPEC.split('\n'))
        self.feat_params = scene_functions.get_params_from_config(self.scene_params)
        self.model_dir = os.path.join(self.final_dir, 'model')
        assert (not os.path.exists(self.model_dir)), "Model folder already exists"
        os.makedirs(self.model_dir)
        self.labels = LABELS
        self.target_label_id = Label.by_name(self.scene_params["model_to_db_injection"]["target_label"]).id
        self.min_pos_frames = int(self.scene_params["train_detector_params"]["pos_min_num_frames"])
        self.min_train_neg_frames = int(self.scene_params["train_detector_params"]["neg_train_min_num_frames"])
        self.min_test_neg_frames = int(self.scene_params["train_detector_params"]["neg_test_min_num_frames"])
        self.split_ratio = float(self.scene_params["train_detector_params"]["split_ratio"])
        self.neg_pos_ratio = float(self.scene_params["train_detector_params"]["neg_pos_ratio"])
        self.dataset = None

    def set_model_files(self, model_dir, test_info, pca_file, vocab_file, svm_file):
        assert (os.path.exists(model_dir)), "Model folder doesn't exists"
        self.model_dir = model_dir
        assert (os.path.exists(test_info)), "test infofile doesn't exists"
        self.test_infofile = test_info
        assert (os.path.exists(pca_file)), "pca file doesn't exists"
        self.pca_filename = pca_file
        assert (os.path.exists(vocab_file)), "vocabulary file doesn't exists"
        self.vocab_filename = vocab_file
        assert (os.path.exists(svm_file)), "svm file doesn't exists"
        self.svm_model_filename = svm_file

    def write_negative_examples(self, false_pos=None):
        """ Get frames from different videos that are not from that label
            Args:
                false_pos = string, path to file with negative examples v\tt
            Returns:
                negfile = string, path to file with negative examples v\tt
        """
        max_videos = int(round(self.scene_params["url_injection"]["max_vids"] * self.neg_pos_ratio))
        images = get_negative_examples(self.target_label_id, max_videos, IMGS, EVERY_N_SEC)
        negfile = os.path.join(self.final_dir, "negative_examples.txt")
        extra = scene_functions.read_image_file(false_pos) if false_pos else []
        images = set(images).union(set(extra))
        logger.info("Total of %d images for negative set" % len(images))
        with open(negfile, 'w') as fo:
            for v, t in images:
                line = "%012d\t%012d\n" % (v, t)
                fo.write(line)
        return negfile

    def get_positive_examples_mturk(self):
        query = session.query(MTurkImage.video_id, MTurkImage.timestamp).filter_by(label_id=self.target_label_id, result=True)
        pos_data = os.path.join(self.final_dir, "positive_examples.txt")
        with open(pos_data, 'w') as fo:
            for v, t in query:
                line = "%012d\t%012d\n" % (v, t)
                fo.write(line)
        return pos_data

    def create_dataset_partitions(self, pos_dataset, neg_dataset):
        """ Create dataset from files
            Args:
                pos_dataset: path to file with positive frames
                neg_dataset: path to file with neg frames
            Assertions:
                AssertionErrors: if files don't exist
        """
        scenedata = SceneDataset(self.final_dir, self.labels)
        scenedata.create_dataset(pos_dataset, neg_dataset, self.split_ratio, self.neg_pos_ratio)
        self.dataset = scenedata
        self.train_infofile = scenedata.infofile["train"]
        self.test_infofile = scenedata.infofile["test"]

    def train_classifier(self):
        """ Train scene classifier """
        scene_functions.check_infofile(self.train_infofile)
        assert self.dataset,  "No dataset found"
        temp_dir = tempfile.mkdtemp()
        extract_filename = os.path.join(temp_dir, 'extract.xml')
        self.pca_filename = os.path.join(self.model_dir, 'pca.xml')
        projected_filename = os.path.join(temp_dir, 'projected.xml')
        self.vocab_filename = os.path.join(self.model_dir, 'vocab.xml')
        hist_filename = os.path.join(temp_dir, 'hist.xml')
        self.svm_model_filename = os.path.join(self.model_dir, 'model.svm')
        data = self.dataset.dataset
        pos = data['pos'].num_train + data['pos'].num_test

        if pos < self.min_pos_frames:
            raise ValueError("Found only %d out of %d positive frames expected" %(pos, self.min_pos_frames))

        if data['neg'].num_train < self.min_train_neg_frames:
            raise ValueError("Found only %d out of %d train negative frames expected" %(data['neg'].num_train, self.min_train_neg_frames))

        if data['neg'].num_test < self.min_test_neg_frames:
            raise ValueError("Found only %d out of %d test negative frames expected" %(data['neg'].num_train, self.min_train_neg_frames))

        bovw_functions.feature_extract(self.bin_dir, self.train_infofile, extract_filename, self.feat_params)
        bovw_functions.pca_computation(self.bin_dir, extract_filename, self.feat_params['pca_dimensions'], self.pca_filename)
        bovw_functions.pca_projection(self.bin_dir, extract_filename, self.pca_filename, projected_filename)
        bovw_functions.vocab_kms(self.bin_dir, projected_filename, self.feat_params['vocab_size'], self.vocab_filename)
        bovw_functions.hist_kms(self.bin_dir, self.vocab_filename, projected_filename, hist_filename)
        bovw_functions.svm_model(hist_filename, self.feat_params['svm_type'], self.feat_params['svm_kernel'], self.svm_model_filename)
        try:
            shutil.copy(self.train_infofile, self.final_dir)
        except shutil.Error:
            pass # They're probably already in the final directory

        shutil.rmtree(temp_dir)

    def test_classifier(self):
        """ Perform testing for the classifier """
        scene_functions.check_infofile(self.test_infofile)
        temp_dir = tempfile.mkdtemp()
        test_extract_filename = os.path.join(temp_dir, 'test_extract.xml')
        test_projected_filename = os.path.join(temp_dir, 'test_projected.xml')
        test_hist_filename = os.path.join(temp_dir, 'test_hist.xml')

        bovw_functions.feature_extract(self.bin_dir, self.test_infofile, test_extract_filename, self.feat_params)
        bovw_functions.pca_projection(self.bin_dir, test_extract_filename, self.pca_filename, test_projected_filename)
        bovw_functions.hist_kms(self.bin_dir, self.vocab_filename, test_projected_filename, test_hist_filename)
        results = bovw_functions.predict(test_hist_filename, self.svm_model_filename)
        try:
            shutil.copy(self.test_infofile, self.final_dir)
        except shutil.Error:
            pass # They're probably already in the final directory
        # Remove intermediate directory
        shutil.rmtree(temp_dir)
        return results

    def evaluate_classifier(self, results):
        data =  self.dataset
        assert data, "No dataset found"
        vids = data.dataset['pos'].dsets['test']
        vids = vids.tolist()
        assert len(vids) , "No positive test data found"
        tp = 0.0
        fp = 0.0
        pos = data.dataset['pos'].num_test
        negs = data.dataset['neg'].num_test
        false_pos = []
        for (v, t), l in results.items():
            if l >= 0.0:
                if [int(v), int(t)] in vids:
                    tp += 1
                else:
                    false_pos.append([int(v), int(t)])
                    fp += 1
        self.fpr = fp / negs
        self.tpr = tp / pos
        return false_pos

    def inject_classifier(self):
        config_path = os.path.join(self.model_dir, 'params.cfg')
        shutil.copy(self.config_file, config_path)
        secs_since_epoc = int((datetime.datetime.now() - datetime.datetime(1970,1,1)).total_seconds())
        det_name = Label.get(self.target_label_id).name + '-%d' % secs_since_epoc
        detector_label = Label.get_or_create(det_name)
        # create detector
        det = SceneDetector(name=det_name)
        det.pca_dimensions = int(self.feat_params['pca_dimensions'])
        det.feature_type = self.feat_params['in_feature_type']
        det.image_threshold = float(self.scene_params['train_detector_params']['image_threshold'])
        det.video_threshold = float(self.scene_params['train_detector_params']['video_threshold'])
        session.flush()
        # add targets for det
        det.add_targets([detector_label])
        di = SceneDetectorInjector(self.model_dir)
        di.tar_and_upload(det)
        self.scene_params["model_to_db_injection"]["detector_id"] = str(det.id)
        self.scene_params.write()
        self.save_training_images(det.id)

    def save_training_images(self, detector_id):
        assert self.dataset, "Dataset is empty"
        for video_id, timestamp in self.dataset.dataset['pos'].dsets['train']:
            if video_id > 0:
                TrainingImage(detector_id=detector_id, video_id=video_id,
                              timestamp=timestamp, label=scene_functions.POS_LABEL)
        session.flush()


class SceneDetectorInjector(AbstractInjector):

    def get_names(self):
        return ['model.svm', 'pca.xml', 'vocab.xml', 'vocab.xml.desc', 'params.cfg']
