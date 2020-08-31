import os

from migrate.versioning.version import Collection
from migrate.versioning import api
from sqlalchemy.engine.base import Connection

from affine import config


_original_execute = Connection.execute
_original_migrate = api._migrate

def monkeypatch_for_migrate():
    """Change the Connection.execute function to split the given SQL text

    on semicolon and execute one statement at a time.
    Otherwise, an exception doesn't get raised if one of the later
    statements fails.
    We use this just for our migrations so errors are correctly raised.
    This isn't smart enough to correctly handle SQL that contains
    literal strings with semicolons in them, etc.
    """
    def execute(self, sql, *args, **kw):
        if not isinstance(sql, basestring):
            return _original_execute(self, sql, *args, **kw)
        # Wrong for quoted strings containing semicolons, etc.
        for stmt in sql.split(';'):
            stmt = stmt.strip()
            if stmt:
                ret = _original_execute(self, stmt, *args, **kw)
        return ret

    def _migrate(*args, **kw):
        check_for_duplicate_versions()
        return _original_migrate(*args, **kw)

    Connection.execute = execute
    api._migrate = _migrate


def unmonkeypatch():
    Connection.execute = _original_execute
    api._migrate = _original_migrate


def check_for_duplicate_versions():
    """Catch cases where two migrations have the same number.

    This probably happened in a merge where two people added a new migration.
    One of them needs to be renumbered.
    """
    repo_dir = os.path.join(config.basedir, 'db', 'versions')
    seen_versions = set()
    for filename in os.listdir(repo_dir):
        if 'default_upgrade' in filename:
            match = Collection.FILENAME_WITH_VERSION.match(filename)
            if match:
                version = int(match.group(1))
                assert version not in seen_versions, 'Duplicate migration #%d' % version
                seen_versions.add(version)
