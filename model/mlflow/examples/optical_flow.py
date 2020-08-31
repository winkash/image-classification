import os
import hashlib
from tempfile import mkdtemp

import numpy as np

from affine.model import *
from affine.detection.model import ColorHistFeatureExtractor, LibsvmClassifier,\
    OpticalFlowFeatureExtractor
from affine.detection.vision.videogame.video_sampler import VideoFFmpegSampler
from affine.detection.model.flow import Step, Flow, FutureLambda, FutureFlowInput
from affine.parallel import pool_map


class VideoAccessObject(object):

    def __init__(self, idx, label, path=None, imagedir=None):
        self.idx = idx
        self.label = label
        self.path = path
        self.imagedir = imagedir
        self.metadata = {}


class ImageAccessObject(object):

    def __init__(self, idx, label=None, video_id=None, timestamp=None, path=None):
        self.idx = idx
        self.label = label
        self.video_id = video_id
        self.timestamp = timestamp
        self.path = path


def convert_to_images(video_objs):
    # Assumes videos are sampled
    image_objs = []
    for v in video_objs:
        for i in os.listdir(v.imagedir):
            image_path = os.path.join(v.imagedir, i)
            # hash/idx could also be videoid_timestamp
            idx = hashlib.sha1(image_path).hexdigest()
            image_objs.append(
                ImageAccessObject(idx=idx, label=v.label, video_id=v.idx, path=image_path))

    return image_objs


def download(args):
    video_id, download_dir = args
    v = Video.get(video_id)
    assert v, video_id
    path = os.path.join(download_dir, str(video_id))
    v.download_video(path)
    return path


def download_videos(video_objs):
    opdir = mkdtemp()
    args = [(video.idx, opdir) for video in video_objs]
    video_paths = pool_map(download, args)
    for obj, path in zip(video_objs, video_paths):
        obj.path = path
    return video_objs


def acquire_data(num_videos):
    query = session.query(Video.id).filter(
        Video.s3_video == True).limit(num_videos * 2)
    data = [VideoAccessObject(row.id, i % 2) for i, row in enumerate(query)]
    return data


def sample(args):
    video_path, op_dir = args
    ss = VideoFFmpegSampler(video_path)
    ss.sample(op_dir)


def sample_videos(video_objs):
    args = []
    for vid in video_objs:
        op_dir = os.path.join(mkdtemp(), str(vid.idx))
        vid.imagedir = op_dir
        args.append((vid.path, op_dir))
    pool_map(sample, args)
    return video_objs


def main():

    color_svm_file = '/tmp/color.svm'
    optical_svm_file = '/tmp/optical.svm'

    f = Flow("Color and optical Classification")
    data_grabber = Step('data', acquire_data, None)
    downloader = Step('download', download_videos, None)
    sampler = Step('sample', sample_videos, 'sample')
    converter = Step('converter', convert_to_images, None)
    color_feature = Step('color', ColorHistFeatureExtractor(), 'extract')
    optical_feature = Step(
        'optical', OpticalFlowFeatureExtractor(), 'extract_multiple')
    color_svm = Step('c_svm', LibsvmClassifier(), 'train')
    optical_svm = Step('o_svm', LibsvmClassifier(), 'train')

    # Adding steps to the pipeline
    for step in [data_grabber, downloader, sampler, converter,
                 color_feature, color_svm, optical_feature, optical_svm]:
        f.add_step(step)

    f.start_with(data_grabber, FutureFlowInput(f, 'num_videos'))
    # Setting up connections
    f.connect(data_grabber, downloader, data_grabber.output)
    f.connect(downloader, sampler, downloader.output)
    f.connect(sampler, converter, sampler.output)
    image_paths = FutureLambda(converter.output, lambda x: [i.path for i in x])
    f.connect(converter, color_feature, image_paths)
    image_labels = FutureLambda(
        converter.output, lambda x: np.asarray([i.label for i in x]))
    f.connect(color_feature, color_svm, color_feature.output, image_labels)
    get_sorted_frames = lambda video_objs: [[os.path.join(d.imagedir, f)
                                             for f in sorted(os.listdir(d.imagedir))]
                                            for d in video_objs]
    sorted_frames = FutureLambda(sampler.output, get_sorted_frames)
    f.connect(sampler, optical_feature, sorted_frames)

    video_labels = FutureLambda(
        sampler.output, lambda video_objs: np.asarray([v.label for v in video_objs]))
    f.connect(optical_feature, optical_svm,
              optical_feature.output, video_labels)

    # Setting up callbacks
    fn = lambda step: step.actor.save_to_file(color_svm_file)
    color_svm.add_done_callback(fn)

    fn = lambda step: step.actor.save_to_file(optical_svm_file)
    optical_svm.add_done_callback(fn)

    # Start pipeline
    f.run_flow(num_videos=2)


if __name__ == '__main__':
    main()
