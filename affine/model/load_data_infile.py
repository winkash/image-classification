"""Helper functions for executing LOAD DATA LOCAL INFILE statements."""

import os
import logging
import shutil
from tempfile import mkdtemp

from affine.model.base import execute
from affine.retries import retry_operation

__all__ = ['load_data_infile']

logger = logging.getLogger(__name__)

DEFAULT_LINES_PER_CHUNK = 5000
DEFAULT_LINE_DELIMITER = '\r\n'

def split_file(path, lines_per_chunk):
    """Divide a file into separate files with lines_per_chunk lines each.

    Return the path of the directory where the new files are.
    """
    outdir = mkdtemp()
    outfile = None
    with open(path, 'rb') as input:
        for line_num, line in enumerate(input):
            if (line_num % lines_per_chunk) == 0:
                if outfile is not None:
                    outfile.close()
                chunk_number = line_num / lines_per_chunk
                filename = '%06d' % chunk_number
                outfile = open(os.path.join(outdir, filename), 'wb')
            outfile.write(line)
    if outfile is not None:
        outfile.close()
    return outdir


def load_chunk_from_file(tablename, path, cols, on_duplicate, post, line_delimiter, **retry_args):
    logger.info("file being loaded: %s", path)
    path = os.path.abspath(path)
    statement = """
        LOAD DATA LOCAL INFILE '%s' %s 
             INTO TABLE `%s` 
             LINES TERMINATED BY '%s'
             (%s)
             %s
    """ % (path, on_duplicate, tablename, line_delimiter, cols, post or '')

    retry_args.setdefault('error_message', 'Failed to execute load statement: %s' % statement)
    retry_operation(execute, statement, **retry_args)


def load_data_infile(tablename, path, cols, on_duplicate, lines_per_chunk=None, line_delimiter=None, post=None, **retry_args):
    path = os.path.abspath(path)
    lines_per_chunk = lines_per_chunk or DEFAULT_LINES_PER_CHUNK
    line_delimiter = line_delimiter or DEFAULT_LINE_DELIMITER
    split_dir = split_file(path, lines_per_chunk=lines_per_chunk)
    try:
        for filename in sorted(os.listdir(split_dir)):
            chunk_file = os.path.join(split_dir, filename)
            load_chunk_from_file(tablename, chunk_file, cols, on_duplicate, post,line_delimiter, **retry_args)
    finally:
        shutil.rmtree(split_dir)
