from datetime import timedelta
import os
import sys
import tempfile

from configobj import ConfigObj

##### Project root directory. It's one directory above this file.
basedir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
##### We store the config settings in private dict, _config
_config = {}


##### Utilities for loading config files

def config_file_path(filename):
    return os.path.join(basedir, 'config', filename)

def config_from_path(path):
    return ConfigObj(path, unrepr=True).dict()['global']

def load_config(path):
    _config.clear()
    _config.update(config_from_path(path))

def load_initial_config():
    filename = 'active.cfg'
    if os.getenv('TEST_MODE') or ('nosetests' in sys.argv[0]):
        filename = 'test.cfg'
    path = config_file_path(filename)
    if os.path.exists(path):
        load_config(path)

load_initial_config()


##### General helper methods for config settings

# Get a config setting
def get(key, default=None):
    return _config.get(key, default)

# Update config settings from dict
def update(settings):
    _config.update(settings)

# Set a config setting
def set(key, value):
    update({key : value})


##### Helper methods for reading specific config settings

def _ensure_dir_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        assert os.path.exists(path), e

def dynamodb_local_dir():
    dynamodb_local_path = get('dynamo.local.path')
    return os.path.join(basedir, dynamodb_local_path)

def bin_dir():
    return os.path.join(basedir, 'bin')

def lib_dir():
    return os.path.join(bin_dir(), 'lib')

def log_dir():
    path = get('affine.log.dir', './log')
    path = os.path.join(basedir, path)
    _ensure_dir_exists(path)
    return path

def scratch_detector_path():
    path = get('affine.detectors.dir')
    assert path
    path = os.path.join(basedir, path)
    _ensure_dir_exists(path)
    return path

def testdata_path(*subpath):
    return os.path.join(basedir, 'testdata', *subpath)

def s3_bucket():
    return get('affine.s3.bucket')

def s3_detector_bucket():
    return get('affine.s3.detector_bucket')

def s3_qa_bucket():
    return get('affine.s3.qa_bucket')

def s3_on_demand_qa_bucket():
    return get('affine.s3.on_demand_qa_bucket')

def sns_qa_topic():
    return get('sns.qa_topic')

def label_checks_topic():
    return get('sns.label_checks_topic')

def mturk_alerts_topic():
    return get('sns.mturk_alerts_topic')

def sns_download_queue_topic():
    return get('sns.download_queue_topic')

def cache_time():
    return get('affine.cache.default_expiration_seconds', 3600)

def qs_li_cache_time():
    return get('affine.query_service.line_item_cache_time', 3600)

def sleep_factor():
    return get('affine.sleep_factor', 1)

def asg_recalc_slave():
    return get('affine.asg.recalculate-slave')

def sqs_recalc_new_queue_name():
    return get('sqs.recalc_new_queue_name')

def sqs_recalc_old_queue_name():
    return get('sqs.recalc_old_queue_name')

def sqs_visibility_timeout():
    return get('sqs.visibility_timeout')

def sqs_text_queue_name():
    return get('sqs.text_queue_name')

def sqs_text_visibility_timeout():
    return get('sqs.text_visibility_timeout')

def sqs_detection_queue_name():
    return get('sqs.detection_queue_name')

def sqs_vcr_detection_queue_name():
    return get('sqs.vcr_detection_queue_name')

def sqs_vcr_text_detection_queue_name():
    return get('sqs.vcr_text_detection_queue_name')

def sqs_download_queue_name():
    return get('sqs.download_queue_name')

def sqs_redis_label_page_queue():
    return get('sqs.redis_label_page_queue')

def mturk_automation_config_path():
    return os.path.join(basedir, 'automation', 'mturk', 'config.json')

def event_log_rotation_increment():
    minutes = get('affine.log.events.rotation_increment', 15)
    return timedelta(minutes=minutes)
