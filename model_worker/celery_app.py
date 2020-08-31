import kombu.transport.redis
from celery import Celery
from affine import config
from affine.detection.model_worker.serializer.custom_pickle import register_pickle
from affine.detection.model_worker.config import BaseConfig

#BUGFIX in kombu for hanging workers
kombu.transport.redis.Channel.socket_timeout = BaseConfig.CELERY_REDIS_SOCKET_TIMEOUT

CELERY_WORKER_ARGS = ['celery', 'worker',
                      '-l', 'info',
                      '-Ofair',]
CELERY_WORKER_TEST_ARGS = CELERY_WORKER_ARGS + ['--concurrency=1']


def configure_app(app):
    test = config.get('celery.test')
    if test:
        from config.celerytestconfig import CeleryTestConfig
        app.config_from_object(CeleryTestConfig)
    else:
        from config.celeryconfig import CeleryConfig
        app.config_from_object(CeleryConfig)


def setup_app():
    register_pickle()
    app = Celery('data_processor',
                 include=['affine.detection.model_worker.'+\
                          'tasks.data_processor_task'])
    return app


celery_app = setup_app()
configure_app(celery_app)
