from logging import getLogger
import os
import cPickle as pickle
import subprocess
import sys
import shutil
import unicodedata
import uuid

from affine import config
from affine.aws import s3client
from affine.detection.model.classifiers import LibsvmClassifier
from affine.model import Label, session, TopicModelDetector, TextDetectionVersion, ClassifierTarget

logger = getLogger(__name__)


class TopicClassifier(object):
    YT = "youtube"

    def __init__(self):
        """ Sets up paths to all binaries, downloads required files from s3, sets up named pipes for categories
        """
        self.server = None
        self.current_version = None

    def start_server(self):
        if self.server is not None:
            return

        self.current_version = TextDetectionVersion.get_latest_version_timestamp(detector_type='topic_model', return_int=True)
        assert self.current_version is not None, "No version!"
        model_file_dir_name = 'classifier_model_files_%d_%s' %(self.current_version, uuid.uuid1())
        self.model_files_dir = os.path.join(config.scratch_detector_path(),
                model_file_dir_name)
        self.grab_s3_files()

        self.load_category_data(os.path.join(self.model_files_dir, "category_info.pickle"))
        self.create_named_pipes()
        self.start_server_process()

    def start_server_process(self):
        cmd = ['java', '-Xmx4096m', '-jar', 'TopicModelFast.jar',
               self.model_files_dir, self.output_pipe, self.input_pipe]
        topic_model_dir = os.path.join(config.bin_dir(), 'topic_model')

        logger.info("starting java server process")
        self.server = subprocess.Popen(cmd, cwd=topic_model_dir)
        logger.info("done starting java server process")

        logger.info("waiting for models to load")
        message = self.receive_message_from_server()
        assert message == ['READY'], message
        logger.info("done waiting for models to load")

    def stop_server(self):
        if self.server is None:
            return
        logger.info("stopping java server process")
        self.server.kill()
        self.server.wait()
        self.server = None
        logger.info("done stopping java server process")

    def restart_server(self):
        self.stop_server()
        self.start_server()

    def configure_server(self):
        latest_version = TextDetectionVersion.get_latest_version_timestamp(detector_type='topic_model', return_int=True)
        if self.server is not None and latest_version == self.current_version:
            return
        logger.info('Restarting server to upgrade to version %d' %latest_version)
        self.restart_server()

    def create_named_pipes(self):
        dir_name = 'named_pipes_%s'%uuid.uuid1()
        named_pipe_dir = os.path.join(config.scratch_detector_path(), dir_name)
        try:
            os.makedirs(named_pipe_dir)
        except OSError as e:
            assert os.path.exists(named_pipe_dir), e
        # We write page text and read back topics
        self.output_pipe = os.path.join(named_pipe_dir, 'text.fifo')
        self.input_pipe = os.path.join(named_pipe_dir, 'topic.fifo')
        os.mkfifo(self.input_pipe)
        os.mkfifo(self.output_pipe)

    def send_message_to_server(self, message):
        """Send message to Java server. Block until it has read"""
        with open(self.output_pipe, "w") as f:
            for line in message:
                f.write(line + "\n")

    def receive_message_from_server(self):
        """Receive message to Java server. Block until it has written"""
        with open(self.input_pipe) as f:
            message = f.read().splitlines()
        if message[0] == 'ERROR':
            error = 'Mallet failed: ' + '\n'.join(message[1:])
            raise Exception(error)
        return message

    def load_category_data(self, category_data_pickle):
        """Loads pickle file containing data structures to map Tier 2 classification labels to categories

           Pickle contains two dictionaries:
           1) category_dict = {0:t1_category1, 1:t1_category2, .....}
           2) category_info:  key = category_name, value = (Tier1_category label name, Tier2_category label name)
           Eg: {..., 'Automotive_Sedan': ('IAB:Automotive', 'IAB:Sedan'), ...}
           The Tier2 classification label for any Tier2 category is its index value in the sorted list of keys
        """
        with open(category_data_pickle, "rb") as f:
            self.category_dict, self.category_info = pickle.load(f)

    def grab_s3_files(self):
        bucket = config.s3_detector_bucket()
        tar_ball_name = 'classifier_model_files_%d'%self.current_version
        logger.info("downloading files from S3")
        s3client.download_tarball(bucket, tar_ball_name, self.model_files_dir)
        logger.info("done downloading files from S3")

    def infer_label_from_text(self, text_string, category):
        """ Infers a label based on given text and category name

        Input:
        text_string: Text string to classify
        category: "youtube" or Tier1 category

        Returns: Predicted label id
        """
        feature_vector = self.mallet_infer_topics(text_string, category)
        if category == TopicClassifier.YT:
            return self.manual_predict(feature_vector, category)

        return self.svm_predict(feature_vector, category)

    def mallet_infer_topics(self, text_string, category):
        """Communicate with Mallet java server to run Tier1 or Tier2 topic modeling

        Input:
        text_string: Text to infer topics from
        category: "youtube" or Tier1 category

        Returns: feature vector, as dict mapping topic IDs to values
        """
        logger.info("talking to java server process")
        message = [category, text_string]
        self.send_message_to_server(message)
        mallet_output = self.receive_message_from_server()
        logger.info("done talking to java server process")
        return self.parse_mallet_output(mallet_output)

    def parse_mallet_output(self, mallet_output):
        """Construct a feature vector from the raw output given by mallet"""
        d = mallet_output[1].split()
        return {int(t) : float(v)
                for t, v in zip(d[2::2], d[3::2])}

    def manual_predict(self, feature_vector, category):
        """Do prediction using human matched topics"""
        topic_category_pickle = os.path.join(self.model_files_dir,
                                             "%s.topic_map" % category)
        with open(topic_category_pickle,"rb") as f:
            topic_category_dict = pickle.load(f)
        sorted_topics = feature_vector.items()
        sorted_topics.sort(key = lambda x:x[1], reverse=True)
        for topic, proportion in sorted_topics:
            if proportion < 0.09:
                return None
            category = topic_category_dict.get(topic)
            if category is not None:
                return category
        return None

    def svm_predict(self, feature_vector, category):
        """Do prediction using trained SVM"""
        model_file = os.path.join(self.model_files_dir, "%s.svm_model" % category)
        clf = LibsvmClassifier.load_from_file(model_file)
        return clf.predict([feature_vector])[0]

    def remove_control(self, text):
        """ Remove unicode control characters from input text and returns clean text
        """
        tbl = {i : None for i in xrange(sys.maxunicode)
            if unicodedata.category(unichr(i)) in ('Zp','Zl','Cf','Cc')}
        if not isinstance(text, unicode):
            text = text.decode('utf-8')
        return text.translate(tbl).encode('utf-8')

    def process_text(self, text_string):
        """ High level function that takes a text string and returns classification

        Input:
        text_string: Text string to classify

        Returns: Tuple (Tier1 category, Tier2 category)
        """
        text_string = self.remove_control(text_string)
        if text_string == "":
            return (None,None)
        tier1_label_id = self.infer_label_from_text(text_string, TopicClassifier.YT)
        if tier1_label_id is None:
            return (None,None)
        tier1_category = self.category_dict[tier1_label_id]
        tier2_label_id = self.infer_label_from_text(text_string, tier1_category)
        return self.category_info[tier2_label_id]

    def process_page(self, page):
        logger.info("Assessing Topic Models for Page: %s" %page.id)
        self.configure_server()
        label1, label2 = self.process_text(page.title_and_text)
        matching_label_ids = []

        ctl, tmd = ClassifierTarget, TopicModelDetector
        for label_id in label1, label2:
            if label_id is not None:

                query = tmd.query.join(tmd.clf_targets)
                query = query.filter_by(target_label_id=label_id)
                det = query.one()

                assert det is not None, 'No detector found for label %d' % label_id
                det.save_result(page.id)
                matching_label_ids.append(label_id)

        query = session.query(tmd.id)
        if matching_label_ids:
            query = query.join(ctl).filter(~ctl.target_label_id.in_(matching_label_ids))
        detector_ids_to_delete = {id for (id,) in query}
        TopicModelDetector.delete_detector_results(page, detector_ids_to_delete)
