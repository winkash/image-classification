"""Database session and the base class for model classes"""
from logging import getLogger
import threading

import flask
import _mysql
from MySQLdb.converters import conversions
from sqlalchemy import engine_from_config, MetaData
from sqlalchemy.engine import url
from sqlalchemy.orm import create_session, scoped_session
from sqlalchemy.exc import ResourceClosedError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import ResourceClosedError
try:
    from sqlalchemy.ext.declarative import _declarative_constructor
except ImportError:
    from sqlalchemy.ext.declarative.api import _declarative_constructor
from savalidation import ValidationMixin

from affine import config

__all__ = [
    'metadata', 'recreate_engines', 'flush_and_close', 'Base',
    'session', 'migrate_session',
    'execute', 'migrate_execute',
]

logger = getLogger(__name__)
CONFIG_ERROR_STR = 'Your database settings are not correctly configured. Do you have a config file selected?'


def scopefunc():
    thread_id = id(threading.current_thread())
    app_name = flask.current_app.name if flask.current_app else None
    return (thread_id, app_name)


def create_primary_session():
    engine = metadata.bind
    assert engine is not None, CONFIG_ERROR_STR

    session = create_session(bind=engine, autoflush=False)
    app = flask.current_app
    if app:
        try:
            configure_session = app.configure_session
        except AttributeError:
            pass
        else:
            configure_session(session)
    return session


def create_migrate_session():
    assert migrate_engine is not None, CONFIG_ERROR_STR
    return create_session(bind=migrate_engine, autoflush=False)


def recreate_engines():
    global migrate_engine

    # Destroy old engines
    for _engine in [metadata.bind, migrate_engine]:
        if _engine is not None:
            # Nuke connection state
            _engine.dispose()

    # Get config for engines
    cfg = config._config.copy()
    migrate_url = cfg.pop('sqlalchemy.master.migrate_url', None)
    # Create primary engine
    if cfg.get('sqlalchemy.master.url'):
        metadata.bind = engine_from_config(cfg, prefix='sqlalchemy.master.')
    # Create new migrate engine
    if migrate_url:
        cfg['sqlalchemy.master.url'] = migrate_url
        migrate_engine = engine_from_config(cfg, prefix='sqlalchemy.master.')


session = scoped_session(create_primary_session, scopefunc=scopefunc)
metadata = MetaData() # metadata.bind is the primary engine
migrate_engine = None
migrate_session = scoped_session(create_migrate_session)
recreate_engines()


def flush_and_close():
    try:
        session.flush()
    except Exception:
        logger.exception('error while flushing')
    session.close()


def execute(query, *args, **kwargs):
    _session = kwargs.pop('session', session)
    result = _session.execute(query, *args, **kwargs)
    if result.closed:
        return
    try:
        return result.fetchall()
    except ResourceClosedError:
        return


def migrate_execute(query, *args, **kwargs):
    kwargs['session'] = migrate_session
    return execute(query, *args, **kwargs)


class Base(ValidationMixin):
    """Base class for ORM objects"""
    query = session.query_property()
    auto_add = True

    @classmethod
    def get(cls, key):
        return cls.query.get(key)

    def __unicode__(self):
        namestr = ''
        if hasattr(self, 'name') and self.name:
            namestr = ' ' + self.name
        elif hasattr(self, 'text') and self.text:
            namestr = ' ' + self.text[:100]
        return u'<%s(%s)%s>' % (self.__class__.__name__, self.id, namestr)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __repr__(self):
        return str(self)

    @classmethod
    def like(cls, name):
        return cls.query.filter(cls.name.like("%%%s%%"%name)).first()

    @classmethod
    def all_like(cls, name):
        return cls.query.filter(cls.name.like("%%%s%%"%name)).all()

    @classmethod
    def by_name(cls, name):
        return cls.query.filter_by(name = name).first()

    @classmethod
    def get_or_create(cls, name):
        return (cls.query.filter_by(name=name).first() or
                cls.create(name=name))

    @classmethod
    def _load_from_file(cls, path, cols, on_duplicate, lines_per_chunk=None, line_delimiter=None, post=None, **retry_args):
        from affine.model.load_data_infile import load_data_infile
        load_data_infile(cls.__tablename__, path, cols, on_duplicate, lines_per_chunk, line_delimiter=line_delimiter, post=post, **retry_args)

    @classmethod
    def create(cls, **kw):
        obj = cls(**kw)
        session.add(obj)
        session.flush()
        return obj

    def __init__(self, **kw):
        _declarative_constructor(self, **kw)
        if Base.auto_add:
            session.add(self)

    @property
    def errors(self):
        self._sav_validate(self, 'before_flush')
        return self.validation_errors


Base = declarative_base(cls=Base, constructor=Base.__init__, metadata=metadata)
