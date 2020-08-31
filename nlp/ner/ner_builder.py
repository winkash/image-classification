import os
import socket
import subprocess
import time
from collections import Counter
from logging import getLogger

import ner
from configobj import ConfigObj
from validate import Validator

from affine import config
from affine.model import NamedEntity, Label, session
from affine.external.crawl.freebase_vocab import FreebaseFilter
from affine.retries import retry_operation

MAX_ENTITY_LENGTH = 50
MIN_ENTITY_LENGTH = 4
MAX_ENTITY_WORDS = 3
NER_PORT = 9090
NO_PROCESS_ERROR = 3
SERVER_CHECK_ATTEMPTS = 15
SERVER_CHECK_INTERVAL = 5

logger = getLogger(__name__)

class EntityExtractor(object):
    def __init__(self):
        self.server = None

    def start_ner_server(self):
        if self.server is not None:
            return
        ner_dir = os.path.join(config.bin_dir(), 'ner')
        cmd = ['java', '-mx1000m', '-cp', 'stanford-ner.jar', 'edu.stanford.nlp.ie.NERServer', '-loadClassifier', 'classifiers/english.conll.4class.caseless.distsim.crf.ser.gz', '-port', str(NER_PORT), '-outputFormat', 'inlineXML']
        logger.info("Starting NER server process")
        self.server = subprocess.Popen(cmd, cwd=ner_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._wait_for_server()

    def _wait_for_server(self):
        try:
            retry_operation(self._check_ner_server, error_class=socket.error,
                            num_tries=SERVER_CHECK_ATTEMPTS, sleep_time=SERVER_CHECK_INTERVAL,
                            error_message='NER server not ready yet',
                            with_traceback=False)
            return
        except socket.error:
            # stop server subprocess and raise exception
            self.stop_ner_server()
            raise Exception('Timed out waiting for NER server to come up')

    def _check_ner_server(self):
        test_text = u'Kobe Bryant plays for LA Lakers'
        tagger = ner.SocketNER(host='localhost', port=NER_PORT)
        return tagger.get_entities(test_text)

    def stop_ner_server(self):
        if self.server is None:
            return
        logger.info("stopping NER server process")
        try:
            self.server.kill()
            self.server.wait()
        except OSError, err:
            if err.errno != NO_PROCESS_ERROR:
                raise
        self.server = None
        logger.info("done stopping NER server process")

    def get_candidates(self, corpus):
        try:
            self.start_ner_server()
            tagger = ner.SocketNER(host='localhost', port=NER_PORT)
            person_fd = Counter()
            org_fd = Counter()
            logger.info('Getting candidates with NER')
            with open(corpus) as fi:
                for lnum, ll in enumerate(fi):
                    entity_dict = tagger.get_entities(ll.strip())
                    person_fd.update({i.lower() for i in entity_dict.get(u'PERSON', [])
                        if MIN_ENTITY_LENGTH <= len(i) <= MAX_ENTITY_LENGTH and len(i.split()) <= MAX_ENTITY_WORDS})
                    org_fd.update({i.lower() for i in entity_dict.get(u'ORGANIZATION', [])
                        if len(i) <= MAX_ENTITY_LENGTH and len(i.split()) <= MAX_ENTITY_WORDS})
                    if lnum%1000 == 0:
                        logger.info("Line %d"%lnum)
            person_candidates = {ne for ne in person_fd if person_fd[ne] > 1}
            org_candidates = {ne for ne in org_fd if org_fd[ne] > 1}

            logger.info('Person candidates: %d, Org candidates: %d'%(len(person_candidates), len(org_candidates)))
            return person_candidates, org_candidates
        finally:
            self.stop_ner_server()


class EntityRunner(object):
    CFG_SPEC = """
    corpus = string
    label_id = integer
    domain = string(default='/common')
    fb_person = string(default='/people/person')
    person_notable = string(default='.*')
    fb_org = string(default='/organization/organization')
    org_notable = string(default='.*')
    """

    @classmethod
    def validate_config_file(cls, config_file):
        config_obj = ConfigObj(config_file, configspec=cls.CFG_SPEC.split('\n'))
        validator = Validator()
        result =  config_obj.validate(validator, copy=True, preserve_errors=True)
        if result != True:
            msg = 'Config file validation failed: %s'%result
            raise Exception(msg)
        ff = FreebaseFilter()
        assert ff.is_domain(config_obj['domain']), "Invalid freebase domain: %s"%config_obj['domain']
        assert ff.is_type(config_obj['fb_person']), "Invalid freebase type: %s"%config_obj['fb_person']
        assert ff.is_type(config_obj['fb_org']), "Invalid freebase type: %s"%config_obj['fb_org']
        assert Label.get(config_obj['label_id']), "Invalid label id: %s"%config_obj['label_id']
        return config_obj

    @classmethod
    def ingest_new_entities(cls, entity_mid_list, label_id,  entity_type):
        assert entity_type in ('person', 'organization'), 'Invalid entity_type: %s'%entity_type
        old_count = NamedEntity.query.filter_by(label_id=label_id, entity_type=entity_type).count()
        for ne, mid in entity_mid_list:
            NamedEntity.get_or_create(ne, entity_type, label_id, fb_id=mid)
        new_count = NamedEntity.query.filter_by(label_id=label_id, entity_type=entity_type).count()
        logger.info('Successfully inserted %d new entities into DB'%(new_count - old_count))

    @classmethod
    def remove_db_entities(cls, entities, label_id, entity_type):
        ''' Takes in a list of named entities and returns the ones not found in the DB'''
        assert entity_type in ('person', 'organization'), 'Invalid entity_type: %s'%entity_type
        db_entities = {i for (i,) in session.query(NamedEntity.name).filter_by(entity_type=entity_type, label_id=label_id)}
        return {ne for ne in entities if ne not in db_entities}

    @classmethod
    def filter_unigrams(cls, candidate_set):
        ff = FreebaseFilter()
        filtered_set = set()
        for ne, fb_id in candidate_set:
            if len(ne.split()) > 1:
                filtered_set.add((ne, fb_id))
            elif ff.filter_unigram(ne, fb_id):
                filtered_set.add((ne, fb_id))
        return filtered_set

    @classmethod
    def run_pipeline(cls, config_file):
        try:
            config_obj = cls.validate_config_file(config_file)
            label_id = config_obj['label_id']
            ee = EntityExtractor()
            person_candidates, org_candidates = ee.get_candidates(config_obj['corpus'])
            person_candidates = cls.remove_db_entities(person_candidates, label_id, 'person')
            org_candidates = cls.remove_db_entities(org_candidates, label_id, 'organization')
            tt = FreebaseFilter.filter_freebase(person_candidates, config_obj['domain'],
                    config_obj['fb_person'], config_obj['person_notable'])
            #only filter unigrams for person entities
            person_new = cls.filter_unigrams(tt)
            org_new = FreebaseFilter.filter_freebase(org_candidates, config_obj['domain'],
                    config_obj['fb_org'], config_obj['org_notable'])
            cls.ingest_new_entities(person_new, label_id, 'person')
            cls.ingest_new_entities(org_new, label_id, 'organization')
        except Exception, err:
            logger.exception('Failed: %s'%err)
