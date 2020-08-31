from affine import config
from affine.aws.elasticache import redis_hosts
from affine.detection.model_worker.config import BaseConfig


class CeleryConfig(BaseConfig):
    REDIS = redis_hosts(config.get('celery.cache_cluster'), sleep_time=0, num_tries=0)[0]
    PORT = config.get('celery.cache_port')
    BROKER_URL = 'redis://guest@{}:{}'.format(REDIS, PORT)
    CELERY_RESULT_BACKEND = 'redis://guest@{}:{}'.format(REDIS, PORT)
    CELERY_ALWAYS_EAGER = False
