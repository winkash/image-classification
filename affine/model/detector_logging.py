import csv
import logging
import os
import tempfile
import uuid

from affine import config
from affine.log import set_handler

__all__ = ['detector_log', 'flush_detector_log_to_db', 'reset_detector_log']


## Globals for detector results logging
logger = None
_log_path = None


def _configure_detector_logging():
    """Set the logfile to be used for detector results logging.

    Can either be file relative to config.log_dir() or
    an absolute path.
    """
    global logger, _log_path
    log_dir = config.log_dir()
    try:
        os.makedirs(log_dir)
    except OSError as e:
        assert os.path.exists(log_dir), e
    filename = 'detector_log-%d.log' % os.getpid()
    _log_path = os.path.join(log_dir, filename)
    # Logger name does not start with affine. so that detection results
    # don't also go to our default log file
    logger = logging.getLogger('detection.logging')
    handler = logging.FileHandler(_log_path)
    set_handler(logger, handler)


def detector_log(*args):
    """Record a line to the detector log file."""
    if logger is None:
        _configure_detector_logging()
    message = '\t'.join(str(arg) for arg in args)
    logger.info(message)


def reset_detector_log():
    if _log_path is not None and os.path.exists(_log_path):
        os.unlink(_log_path)
    _configure_detector_logging()


def flush_detector_log_to_db():
    from affine.model import (ImageDetectorResult, BoxDetectorResult, VideoDetectorResult,
                              TextDetectorResult, TextBoxResult)

    if _log_path is None or not os.path.exists(_log_path):
        return

    random_fname = str(uuid.uuid4()) + '.csv'
    scratch = tempfile.gettempdir()
    vdr_path = os.path.join(scratch, 'VDR_' + random_fname)
    idr_path = os.path.join(scratch, 'IDR_' + random_fname)
    bdr_path = os.path.join(scratch, 'BDR_' + random_fname)
    tdr_path = os.path.join(scratch, 'TDR_' + random_fname)
    tbr_path = os.path.join(scratch, 'TBR_' + random_fname)

    try:
        vdr_file = open(vdr_path, "w")
        idr_file = open(idr_path, "w")
        bdr_file = open(bdr_path, "w")
        tdr_file = open(tdr_path, "w")
        tbr_file = open(tbr_path, "w")

        vdr_csv = csv.writer(vdr_file, delimiter="\t")
        idr_csv = csv.writer(idr_file, delimiter="\t")
        bdr_csv = csv.writer(bdr_file, delimiter="\t")
        tdr_csv = csv.writer(tdr_file, delimiter="\t")
        tbr_csv = csv.writer(tbr_file, delimiter="\t")

        with open(_log_path) as _file:
            for line in _file:
                cols = [col.strip() for col in line.split('\t')]
                row_type = cols[0]
                records = cols[1:]

                if row_type == "IDR":
                    idr_csv.writerow(records)
                elif row_type == "BDR":
                    bdr_csv.writerow(records)
                elif row_type == "TDR":
                    tdr_csv.writerow(records)
                elif row_type == "VDR":
                    vdr_csv.writerow(records)
                elif row_type == "TBR":
                    tbr_csv.writerow(records)
                else:
                    raise TypeError("Unknown log type %s" % row_type)

        idr_file.close()
        bdr_file.close()
        tdr_file.close()
        vdr_file.close()
        tbr_file.close()

        ImageDetectorResult.load_from_file(idr_path)
        BoxDetectorResult.load_from_file(bdr_path)
        TextDetectorResult.load_from_file(tdr_path)
        VideoDetectorResult.load_from_file(vdr_path)
        TextBoxResult.load_from_file(tbr_path)
    finally:
        for file in [idr_path, bdr_path, tdr_path, vdr_path, tbr_path]:
            if os.path.exists(file):
                os.unlink(file)
        reset_detector_log()
