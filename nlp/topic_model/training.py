import argparse
import ast
import os
import shutil
import sys

from configobj import ConfigObj
from logging import getLogger
from tempfile import mkdtemp
from validate import ValidateError, Validator

from affine.detection.nlp.topic_model import *

logger = getLogger(__name__)

__all__ = ['PipelineRunner']


class PipelineRunner(object):
    CFG_SPEC = """
    target_label_name = string
    detector_name = string
    query_dict = yt_query()
    test_split = float(min=0.0, max=1.0, default=0.1)
    neg_tarball_s3 = string(default='neg_json.tar.gz')
    pos_train_json = string(default='pos_train_json')
    pos_test_json = string(default='pos_test_json')
    neg_train_json = string(default='neg_train_json')
    neg_test_json = string(default='neg_test_json')
    youtube_data_ignore_labels = list(default=list())
    include_related = boolean(default=False)
    general_vocab = string(default='general_vocab')
    stop_file = string(default='stop_file')
    pipe_file = string(default='pipe_file')
    vocab_file = string(default='vocab_file')
    model_stats = string(default='model_stats')
    topic_thresholds = topic_thresholds(default='[]')
    [mallet_import]
        input = string(default='mallet_input_file')
        line-regex = string(default='^(.*)$')
        label = integer(default=0)
        name = integer(default=0)
        data = integer(default=1)
        keep-sequence = boolean(default=True)
        output = string(default='tmp.mallet')
    [mallet_train]
        input = string(default='tmp.mallet')
        num-topics = integer(min=1, max=2000, default=500)
        num-iterations = integer(min=1, max=2000, default=400)
        output-topic-keys = string(default='topics')
        output-doc-topics = string(default='output_doc_topics')
        doc-topics-max = integer(default=5)
        inferencer-filename = string(default='inferencer_file')
    [classifier_params]
        bin_thresh = float(min=0.0, max=1.0, default=0.09)
        model_file = string(default='model_file')
        libsvm_file = string(default='libsvm_file')"""

    CFG_NAME = 'lda.cfg'

    @classmethod
    def validate_config_file(cls, config_file):
        config_dict = ConfigObj(config_file, configspec=cls.CFG_SPEC.split('\n'))
        validator = Validator({'yt_query':cls.query_check, 'topic_thresholds':cls.topic_thresholds_check})
        result =  config_dict.validate(validator, copy=True, preserve_errors=True)
        if result != True:
            msg = 'Config file validation failed: %s'%result
            raise Exception(msg)
        for t, _ in config_dict['topic_thresholds']:
            assert 0 <= t <= config_dict['mallet_train']['num-topics'], 'Topic thresholds error!'
        return config_dict

    @classmethod
    def query_check(cls, value):
        try:
            query_dict = ast.literal_eval(value)
            assert len(query_dict)
            for i in query_dict:
                assert isinstance(i, basestring)
                assert isinstance(query_dict[i], int)
                assert 0 < query_dict[i] <= YoutubeCollector.YT_MAX_LIMIT
            return query_dict
        except:
            raise ValidateError("%s invalid dictionary!"%value)
        return value

    @classmethod
    def topic_thresholds_check(cls, value):
        try:
            topic_thresholds = ast.literal_eval(value)
            assert isinstance(topic_thresholds, list)
            for t, v in topic_thresholds:
                assert isinstance(t, int)
                assert isinstance(v, float)
                assert 0.0 <= v <= 1.0
            return topic_thresholds
        except:
            raise ValidateError("%s invalid list of topic thresholds!"%value)
        return value

    @classmethod
    def run_pipeline(cls, config_file):
        config_file = os.path.abspath(config_file)
        logger.info("checking config file")
        config_dict = cls.validate_config_file(config_file)
        model_dir = mkdtemp()   #User's responsibility to delete the directory
        cwdir = os.getcwd()
        os.chdir(model_dir)
        try:
            logger.info("Model files directory: %s"%model_dir)
            logger.info('INGESTING DATA')
            create_traintest_json(config_dict)
            tt = TopicTrainer(config_dict)
            logger.info('TRAINING')
            tt.train_tm()
            tt.train_classifier()
            logger.info('TESTING')
            tt.check_model()
            shutil.copyfile(config_file, cls.CFG_NAME)
            logger.info('DONE!')
            logger.info("Model files directory: %s"%model_dir)
            return model_dir
        except Exception:
            raise
        finally:
            os.chdir(cwdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config-file', dest='config_file', required=True,
        help='config file path')
    args = parser.parse_args()
    if not os.path.exists(args.config_file):
        logger.error('Config file does not exist!')
        sys.exit(1)
    PipelineRunner.run_pipeline(args.config_file)
