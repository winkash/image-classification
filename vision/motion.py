#!/usr/bin/env python
from contextlib import contextmanager
import os
import tempfile
from uuid import uuid1

import numpy as np

from affine import config
from affine.parallel import pool_map
from affine.video_processing.tools import run_cmd
from affine.detection.model.classifiers import LibsvmClassifier

FFMPEG_CMD = '%(bin_dir)s/ffmpeg -i %(video)s -affine_hist %(histfile)s \
             -affine_cls %(video_cls)s'

@contextmanager
def delete_file(filename):
    """Try to delete a file option completion of the block"""
    try:
        yield
    finally:
        try:
            os.unlink(filename)
        except OSError:
            pass


def scratch_path(extension=''):
    """Create path for a temp file/ dir with a given file extension."""
    scratch = tempfile.gettempdir()
    return os.path.join(scratch, str(uuid1()).replace('-', '')) + extension


def video_hist(video_path, video_cls = -1):
    """Calculate motion vector histogram of video"""
    histfile = scratch_path('.hist')
    with delete_file(histfile):
        video_path = os.path.abspath(video_path)
        params = dict(video = video_path,
                      video_cls = video_cls,
                      histfile = histfile,
                      bin_dir = config.bin_dir())
        run_cmd(FFMPEG_CMD, params)
        hist_str = file(histfile).read()
        hist = map(float, hist_str.strip().split(','))
        # first col is label, last 2 cols are meant for (video_id, timestamp)
        # Should be ignored
        return hist[1:-2]


def rate_video(video_path, svmfile):
    """Calculate motion histogram SVM rating of video"""
    hist = video_hist(video_path, -1)
    test_feature = np.asarray([hist])
    clf = LibsvmClassifier.load_from_file(svmfile)
    return clf.predict(test_feature)[0]


def mapped_func(item):
    return video_hist(**item)


def training_data(positive_videos, negative_videos):
    return ([{'video_cls' : -1, 'video_path' : path}
                for path in positive_videos] +
            [{'video_cls' :  1, 'video_path' : path}
                for path in negative_videos])


def train(positive_videos, negative_videos, svmfile):
    """Train a motion histogram SVM from two lists of video paths
       and save it to svmfile.
       Cross-train if no svmfile is given"""
    videos = training_data(positive_videos, negative_videos)
    hists = pool_map(mapped_func, videos)
    features = np.asarray(hists)
    clf = LibsvmClassifier(svm_type=3, kernel_type=5, cache_size=7200)
    labels = np.asarray([item['video_cls'] for item in videos])
    clf.train(features, labels)
    clf.save_to_file(svmfile)
