import cPickle as pickle
import nltk
import os
import re
import tarfile

from configobj import ConfigObj
from logging import getLogger
from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import BernoulliNB
from tempfile import mkstemp

from affine import config
from affine.aws import s3client
from affine.video_processing.tools import run_cmd
from .ingest_training_data import YoutubeVideoText

logger = getLogger(__name__)

__all__ = ['TopicTrainer']


class TopicTrainer(object):

    MALLET_BIN = os.path.join(config.bin_dir(), 'topic_model','mallet')
    RGX_PAT = re.compile(ur'\w\w\w+', re.UNICODE)

    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.n_pos_train = TopicTrainer.file_line_counter(config_dict['pos_train_json'])
        self.n_neg_train = TopicTrainer.file_line_counter(config_dict['neg_train_json'])
        self.n_pos_test = TopicTrainer.file_line_counter(config_dict['pos_test_json'])
        self.n_neg_test = TopicTrainer.file_line_counter(config_dict['neg_test_json'])

    @staticmethod
    def file_line_counter(infile):
        for i, _ in enumerate(open(infile)):
            continue
        return i+1

    @staticmethod
    def create_mallet_config_files(mallet_config_dict):
        h, mallet_config_file = mkstemp(); os.close(h)
        config_obj = ConfigObj(mallet_config_dict)
        config_obj.filename = mallet_config_file
        config_obj.write()
        return mallet_config_file

    @staticmethod
    def write_mallet_input_file(pos_json, neg_json, outfile, vocab_set, include_related=False):
        """ Creates mallet compatible file from json file"""
        with open(outfile, 'w') as fo:
            for ll in open(pos_json):
                jsn = ll.strip()
                one_line =  TopicTrainer.json_to_text_line(jsn, include_related)
                fo.write(TopicTrainer.preprocess_text(one_line, vocab_set).encode('utf-8')+'\n')
            for ll in open(neg_json):
                jsn = ll.strip()
                one_line =  TopicTrainer.json_to_text_line(jsn, include_related)
                fo.write(TopicTrainer.preprocess_text(one_line, vocab_set).encode('utf-8')+'\n')

    def set_vocabulary(self):
        general_vocab = self.config_dict['general_vocab']
        stop_file = self.config_dict['stop_file']
        TopicTrainer.get_resource_file(general_vocab)
        TopicTrainer.get_resource_file(stop_file)
        self.vocab_set = set(open(general_vocab).read().decode('utf-8').splitlines())
        self.stop_set = set(open(stop_file).read().decode('utf-8').splitlines())

        freq_table = {}
        exc_vocab = self.vocab_set | self.stop_set
        # Add vocabulary from positive examples for new category
        for ll in open(self.config_dict['pos_train_json']):
            jsn = ll.strip()
            one_line = TopicTrainer.json_to_text_line(jsn, include_related=self.config_dict['include_related'])
            tokens =  TopicTrainer.tokenize_text(one_line)
            for token in tokens:
                if token not in exc_vocab:
                    freq_table[token] =  freq_table.get(token, 0) + 1
        for token in freq_table:
            if freq_table[token] > 1:
                self.vocab_set.add(token)

    @staticmethod
    def preprocess_text(one_line, vocab_set):
        tokens =  TopicTrainer.tokenize_text(one_line)
        in_vocab = [w for w in tokens if w in vocab_set]
        return ' '.join(in_vocab)

    @staticmethod
    def tokenize_text(one_line):
        pst = nltk.PorterStemmer()
        all_tokens = TopicTrainer.RGX_PAT.findall(one_line.lower())
        stemmed_tokens = []
        for token in all_tokens:
             # only stem plurals
            if token.endswith('s'):
                token = pst.stem(token)
            # 3 letter words can be stemmed down
            if len(token) >=3:
                stemmed_tokens.append(token)
        return stemmed_tokens

    @staticmethod
    def get_resource_file(resource_file):
        bucket = config.s3_detector_bucket()
        tarball_name = resource_file + '.tar.gz'
        s3client.download_from_s3(bucket, tarball_name, tarball_name)
        with tarfile.open(tarball_name, 'r:*') as tar:
            tar.extractall()
        assert os.path.isfile(resource_file)

    @staticmethod
    def json_to_text_line(jsn, include_related=False):
        yvt = YoutubeVideoText.to_object(jsn)
        if include_related:
            st = '\t'.join([yvt.video_title, '%s'%yvt.video_description, ' '.join(['%s'%i for i in yvt.video_comments]), ' '.join(yvt.related_videos_text)])
        else:
            st = '\t'.join([yvt.video_title, '%s'%yvt.video_description, ' '.join(['%s'%i for i in yvt.video_comments])])
        return st

    def train_tm(self):
        logger.info('Setting vocabulary and stopwords')
        self.set_vocabulary()
        logger.info('Converting training json into mallet data')
        TopicTrainer.write_mallet_input_file(self.config_dict['pos_train_json'],
                self.config_dict['neg_train_json'], self.config_dict['mallet_import']['input'],
                self.vocab_set, include_related=self.config_dict['include_related'])
        TopicTrainer.mallet_import_and_train(self.config_dict)
        # write smaller pipe file
        logger.info('Creating vocab file')
        with open(self.config_dict['vocab_file'], 'w') as fo:
            for w in self.vocab_set:
                fo.write(w.encode('utf-8')+'\n')

    @staticmethod
    def mallet_import_and_train(config_dict):
        ''' generic method that takes a config dict as input and runs the LDA algorithm.
        The config dict must contain the keys 'mallet_import' and 'mallet_train'
        for the mallet specific import and training parameters
        '''
        import_config_file = TopicTrainer.create_mallet_config_files(config_dict['mallet_import'])
        logger.info('Running mallet import')
        run_cmd([TopicTrainer.MALLET_BIN,'import-file','--config', import_config_file],
            timeout=None)
        train_config_file = TopicTrainer.create_mallet_config_files(config_dict['mallet_train'])
        logger.info('Running mallet training')
        run_cmd([TopicTrainer.MALLET_BIN,'train-topics','--config', train_config_file],
            timeout=None)
        logger.info('Done training topic models')
        os.unlink(import_config_file)
        os.unlink(train_config_file)
        # write smaller pipe file
        logger.info('Creating pipe file')
        h, tmp_file = mkstemp(); os.close(h)
        run_cmd([TopicTrainer.MALLET_BIN, 'import-file', '--input', tmp_file, '--use-pipe-from',
            config_dict['mallet_import']['output'], '--output', config_dict['pipe_file']],
            timeout=None)
        os.unlink(tmp_file)

    @staticmethod
    def doc_topics_to_libsvm(doc_topics_file, output_file, n_pos):
        ''' Requires that the first n_pos lines are positive examples'''
        fo = open(output_file,"w")
        fi = open(doc_topics_file)
        #skip header
        fi.readline()
        for lnum, l in enumerate(fi):
            ll = l.split()
            a = zip(ll[2::2],ll[3::2])
            a.sort(key = lambda x:int(x[0]))
            b = [x[0]+":"+x[1] for x in a]
            if lnum < n_pos:
                lbl = '1'
            else:
                lbl = '0'
            fo.write('%s %s\n'%(lbl," ".join(b)))
        fi.close()
        fo.close()

    def train_classifier(self):
        libsvm_file = self.config_dict['classifier_params']['libsvm_file']
        TopicTrainer.doc_topics_to_libsvm(self.config_dict['mallet_train']['output-doc-topics'], libsvm_file, self.n_pos_train)
        num_topics = self.config_dict['mallet_train']['num-topics']
        x_train, y_train = load_svmlight_file(libsvm_file, num_topics, zero_based=True)
        bnb = BernoulliNB(binarize=self.config_dict['classifier_params']['bin_thresh'])
        logger.info('Training classifier')
        bnb.fit(x_train, y_train)
        pickle.dump(bnb, open(self.config_dict['classifier_params']['model_file'], "wb"))
        logger.info('Done training classifier')

    def check_model(self):
        h, mallet_input_file = mkstemp(); os.close(h)
        logger.info('Converting testing json into mallet data')
        # Hold out data should never include related text
        TopicTrainer.write_mallet_input_file(self.config_dict['pos_test_json'],
            self.config_dict['neg_test_json'], mallet_input_file, self.vocab_set)
        pipe_file = self.config_dict['pipe_file']
        output = self.config_dict['mallet_import']['output']
        logger.info('Running mallet import')
        run_cmd([TopicTrainer.MALLET_BIN,'import-file','--input', mallet_input_file,
            '--use-pipe-from', pipe_file, '--output', output], timeout=None)

        h, output_doc_topics = mkstemp(); os.close(h)
        infer_config_dict = {}
        infer_config_dict['output-doc-topics'] = output_doc_topics
        infer_config_dict['inferencer'] = self.config_dict['mallet_train']['inferencer-filename']
        infer_config_dict['doc-topics-max'] = self.config_dict['mallet_train']['doc-topics-max']
        infer_config_dict['input'] = output

        infer_config_file = TopicTrainer.create_mallet_config_files(infer_config_dict)
        logger.info('Running mallet inference')
        run_cmd([TopicTrainer.MALLET_BIN,'infer-topics','--config', infer_config_file],
            timeout=None)

        # Check prediction accuracy
        h, libsvm_file = mkstemp(); os.close(h)
        num_topics = self.config_dict['mallet_train']['num-topics']
        TopicTrainer.doc_topics_to_libsvm(output_doc_topics, libsvm_file, self.n_pos_test)
        x_test, _ = load_svmlight_file(libsvm_file, num_topics, zero_based=True)
        h, prediction_file = mkstemp(); os.close(h)
        logger.info('Running classifier prediction')
        # manual matching if topic_thresholds provided
        if len(self.config_dict['topic_thresholds']):
            TopicTrainer.manual_prediction(x_test, self.config_dict['topic_thresholds'], prediction_file)
            self.write_model_stats(prediction_file, model_name='Manually matched')
        else:
            TopicTrainer.model_prediction(x_test, self.config_dict['classifier_params']['model_file'], prediction_file)
            self.write_model_stats(prediction_file)

        os.unlink(mallet_input_file)
        os.unlink(output_doc_topics)
        os.unlink(libsvm_file)
        os.unlink(prediction_file)

    @staticmethod
    def model_prediction(x_test, model_file, prediction_file):
        classifier = pickle.load(open(model_file,"rb"))
        y_pred = classifier.predict(x_test)
        with open(prediction_file, "w") as fo:
            for i in y_pred:
                fo.write('%s\n'%int(i))

    @staticmethod
    def manual_prediction(x_test, topic_thresholds, prediction_file):
        y_pred = []
        for xx in x_test:
            pred = 0
            for t, v in topic_thresholds:
                if xx[0, t] >= v:
                    pred = 1
                    break
            y_pred.append(pred)
        with open(prediction_file, "w") as fo:
            for i in y_pred:
                fo.write('%s\n'%int(i))

    def write_model_stats(self, prediction_file, model_name='Naive Bayes'):
        tp, tn, fp, fn = TopicTrainer.get_acc_numbers(prediction_file, self.n_pos_test)
        precision = float(tp)/((tp + fp) or 1)
        recall = float(tp)/((tp + fn) or 1)
        with open(self.config_dict['model_stats'], 'w') as fo:
            fo.write('%s Model Stats\n'%model_name)
            fo.write('positive training docs = %d\n'%self.n_pos_train)
            fo.write('negative training docs = %d\n'%self.n_neg_train)
            fo.write('positive testing docs = %d\n'%self.n_pos_test)
            fo.write('negative testing docs = %d\n'%self.n_neg_test)
            fo.write('TPs, TNs, FPs, FNs = (%d, %d, %d, %d)\n'%(tp, tn, fp, fn))
            fo.write('Precision = %f\n'%precision)
            fo.write('Recall = %f\n'%recall)

    @staticmethod
    def get_acc_numbers(prediction_file, n_pos):
        tp = tn = fp = fn = 0 # funny shape
        with open(prediction_file) as fi:
            for lnum, ll in enumerate(fi):
                p = int(ll.strip())
                if lnum < n_pos:
                    if p == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if p == 1:
                        fp += 1
                    else:
                        tn += 1
        return tp, tn, fp, fn #shaken not stirred
