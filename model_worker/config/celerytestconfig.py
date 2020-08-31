from affine.detection.model_worker.config import BaseConfig


class CeleryTestConfig(BaseConfig):
    BROKER_URL = 'redis+socket:///tmp/redis.sock'
    CELERY_RESULT_BACKEND = 'redis+socket:///tmp/redis.sock'
    CELERY_ALWAYS_EAGER = True
