import argparse
import cPickle as pickle
import os
import warnings
from collections import defaultdict
from logging import getLogger
from math import ceil
from tempfile import mkstemp
from validate import Validator

from configobj import ConfigObj
from scipy.sparse import csr_matrix, SparseEfficiencyWarning
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import binarize

from affine.model import *
from affine import config
from affine.aws.s3client import download_tarball
from affine.detection.nlp.topic_model import TopicTrainer

logger = getLogger(__name__)

warnings.simplefilter('ignore', SparseEfficiencyWarning)

CHUNK_SIZE = 10000
STOP_THRESH = 0.5
PAGE_TITLE = 'page_title'
PAGE_TEXT = 'page_text'
PAGE_ID = 'page_id'
PAGE_DOMAIN = 'page_domain'

LIBSVM_FILE = 'libsvm_file'
VOCAB_FILE = 'general_vocab'
DOMAIN_WF = 'domain_wf'
TC_FILE = 'tc_file'

__all__ = ['InventoryTopicModeler']

class InventoryTopicModeler(object):

    CFG_SPEC = """
    pipe_file = string(default='pipe_file')
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
        num-threads = integer(default=4)
    """

    def __init__(self, config_file):
        self.config_obj = self.validate_config_file(config_file)

    @staticmethod
    def grab_inventory_text():
        page_ids = sorted([i for (i,) in session.query(WebPageInventory.page_id)])
        n_chunks = ceil(len(page_ids)/float(CHUNK_SIZE))
        with open(PAGE_ID, 'w') as fo1, open(PAGE_TEXT, 'w') as fo2, open(PAGE_TITLE, 'w') as fo3, open(PAGE_DOMAIN, 'w') as fo4:
            for i in xrange(0, len(page_ids), CHUNK_SIZE):
                logger.info('Processing chunk (%d / %d)'%(i/CHUNK_SIZE + 1, n_chunks))
                id_chunk = page_ids[i:i+CHUNK_SIZE]
                query = session.query(WebPage.id, WebPage.title, WebPage.domain).filter(WebPage.id.in_(id_chunk))
                text_dict = web_pages.get_page_text_dict(id_chunk)
                for pg_id, pg_title, pg_domain in query:
                    try:
                        fo1.write('%s\n'%pg_id)
                        fo2.write(text_dict[pg_id].encode('utf-8')+'\n')
                        fo3.write(pg_title.encode('utf-8')+'\n')
                        fo4.write(pg_domain.encode('utf-8')+'\n')
                    except Exception, e:
                        logger.exception('page-id: %s' %pg_id)

    @classmethod
    def validate_config_file(cls, config_file):
        config_obj = ConfigObj(config_file, configspec=cls.CFG_SPEC.split('\n'))
        validator = Validator()
        result =  config_obj.validate(validator, copy=True, preserve_errors=True)
        if result != True:
            msg = 'Config file validation failed: '+str(result)
            raise Exception(msg)
        return config_obj

    def clean_text(self):
        logger.info('Cleaning text...')
        TopicTrainer.get_resource_file(VOCAB_FILE)
        self.fetch_domain_stopwords()
        vocab_set = set(open(VOCAB_FILE).read().decode('utf-8').splitlines())
        with open(self.config_obj['mallet_import']['input'],'w') as fo, open(PAGE_TITLE) as fi1, open(PAGE_TEXT) as fi2, open(PAGE_DOMAIN) as fi3:
            for title, text, domain in zip(fi1, fi2, fi3):
                title_and_text =  title.strip() + ' ' + text.strip()
                clean_text = TopicTrainer.preprocess_text(title_and_text.decode('utf-8'), vocab_set)
                clean_text = self.clean_domain(clean_text, domain.strip())
                fo.write(clean_text.encode('utf-8') + '\n')

    def fetch_domain_stopwords(self):
        self.stop_dict = defaultdict(set)
        download_tarball(config.s3_detector_bucket(), DOMAIN_WF, DOMAIN_WF)
        for pp in os.listdir(DOMAIN_WF):
            pickle_file = os.path.join(DOMAIN_WF, pp)
            with open(pickle_file, 'rb') as fi:
                wf = pickle.load(fi)
            for i in wf:
                if wf[i] > STOP_THRESH:
                    self.stop_dict[pp.replace('.pickle', '')].add(i)

    def clean_domain(self, text, domain):
        clean_text = ' '.join([w for w in text.split(' ') if w not in self.stop_dict[domain]])
        return clean_text

    def train_topic_model(self):
        self.clean_text()
        TopicTrainer.mallet_import_and_train(self.config_obj)
        doc_topics_file = self.config_obj['mallet_train']['output-doc-topics']
        TopicTrainer.doc_topics_to_libsvm(doc_topics_file, LIBSVM_FILE, 0)
        self.create_counts_file()

    @classmethod
    def create_counts_file(cls, libsvm_file=LIBSVM_FILE, n_ftrs=None, counts_file=TC_FILE, topic_threshold=None):
        if n_ftrs is None:
            n_ftrs = cls.validate_config_file(None)['mallet_train']['num-topics']
        x_test, _ = load_svmlight_file(libsvm_file, n_ftrs, zero_based=True)
        if topic_threshold:
            x_test = binarize(x_test, threshold=topic_threshold)
        with open(counts_file,'w') as fo:
            for i in range(n_ftrs):
                col = x_test.getcol(i)
                nzs, _ = col.nonzero()
                fo.write('%d\n'%len(nzs))

    @classmethod
    def create_text_file(cls, topic_id_list, libsvm_file=LIBSVM_FILE, input_file=PAGE_TITLE, n_ftrs=None, topic_threshold=None, op_file='output_trues'):
        if n_ftrs is None:
            n_ftrs = InventoryTopicModeler.validate_config_file(None)['mallet_train']['num-topics']
        x_test, _ = load_svmlight_file(libsvm_file, n_ftrs, zero_based=True)
        if topic_threshold:
            x_test = binarize(x_test, threshold=topic_threshold)
        # column of zeros with length D(# of docs)
        sum_col = csr_matrix((x_test.shape[0], 1), dtype=float)
        for topic_id in topic_id_list:
            sum_col = sum_col + x_test.getcol(topic_id)

        nzs, _ = sum_col.nonzero()
        nzs = set(nzs)
        num_res = 0
        with open(op_file,'w') as fo:
            for i, ll in enumerate(open(input_file)):
                if i in nzs:
                    fo.write('%s\n'%(ll.strip()))
                    num_res += 1
            print 'Number of results = %d'%num_res

    @classmethod
    def estimate_recall(cls, label_id, topic_id_list, page_ids_file=PAGE_ID, libsvm_file=LIBSVM_FILE, n_ftrs=None, topic_threshold=None, missed_file=None):
        h, op_file = mkstemp(); os.close(h)
        cls.create_text_file(topic_id_list, libsvm_file=libsvm_file, input_file=page_ids_file, n_ftrs=n_ftrs, topic_threshold=topic_threshold, op_file=op_file)
        tm_page_ids = {int(i) for i in open(op_file).read().splitlines()}
        # Count wplr True matches for the "true" pages
        query = session.query(WebPageLabelResult.page_id).filter_by(label_id=label_id).filter(WebPageLabelResult.page_id.in_(tm_page_ids))
        wplr_page_ids = {i for (i,) in query}
        missed_page_ids = tm_page_ids - wplr_page_ids
        recall = float(len(wplr_page_ids))/len(tm_page_ids)
        print 'No. of pages from topic modeling: %d'%len(tm_page_ids)
        print 'No. of pages not found in WPLRS: %d'%len(missed_page_ids)
        print 'Approximate Recall: %f'%recall
        if missed_file:
            with open(missed_file, 'w') as fo:
                fo.write('\n'.join([str(i) for i in missed_page_ids]))
        os.unlink(op_file)
        return recall

def run_pipeline(config_file, model_dir):
    cwdir = os.getcwd()
    os.chdir(model_dir)
    try:
        logger.info("Model files directory: %s"%model_dir)
        itm = InventoryTopicModeler(config_file)
        logger.info('Collecting data from inventory')
        itm.grab_inventory_text()
        itm.train_topic_model()
    except Exception:
        raise
    finally:
        os.chdir(cwdir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config-file', dest='config_file', required=False,
        help='config file path')
    parser.add_argument('-model-dir', dest='model_dir', required=False,
        help='working directory where all the model files should go.')
    args = parser.parse_args()
    if args.model_dir is None:
        args.model_dir = mkdtemp()   #User's responsibility to delete the directory
    run_pipeline(args.config_file, args.model_dir)

if __name__ == '__main__':
    main()
