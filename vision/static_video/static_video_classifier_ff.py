"""
Class to train and test classifier for motion-color based video detection
"""
import shutil
import os
import numpy as np

from logging import getLogger
from affine.detection.model.features import OpticalFlowFeatureExtractor
from affine.detection.vision.video_motioncolor.video_sampler import \
    VideoFFmpegSampler
from affine.detection.vision.utils.scene_functions import get_config
from affine.detection.model.mlflow import Step, Flow, FutureFlowInput

__all__ = ['StaticVideoClassifierFlowFactory']

logger = getLogger(__name__)

STATIC_VIDEO_CFG_SPEC = """
    [sampling]
        duration = integer(min=1, max=90, default=30)
        fps = float(min=0.1, max=60, default=1)
        scale = float(min=0.0001, max=4096, default=320)
    [opt_flow]
        masked = boolean(default=False)
        static = boolean(default=True)
        num_bins = integer(min=8, max=32, default=8)
    [classifier]
        slideshow_th = float(default=0)
        video_th = float(min=0.0, max=1.0, default=0.1)
        photo_tl_id = integer(default=9801)
        slideshow_tl_id = integer(default=9802)
    """


class StaticVideoClassifierFlowFactory(object):

    VIDEO, PHOTO, SLIDESHOW, UNKNOWN = [0, 1, 2, 3]
    MIN_FRAMES = 30
    model_files = ['ConfigFile.cfg']

    def __init__(self, config_file):
        self._load_config(config_file)
        self.config_file = config_file
        self.optical_flow_extractor = OpticalFlowFeatureExtractor(
            get_masked_flow=self.opt_flow_masked,
            get_static_flow=self.opt_flow_static,
            numBins=self.opt_flow_num_bins)

    def save_to_dir(self, path):
        """Saves model files to directory.

        Creates directory if it does not exist.

        Args:
            path: Path to directory.
        """
        if not os.path.exists(path):
            os.mkdir(path)
        shutil.copy(os.path.join(self.config_file), path)

    @classmethod
    def load_from_dir(cls, path):
        """Loads model directory.

        Args:
            path: Path to directory.

        Returns:
            A classifier.

        Raises:
            AssertionError: Specified directory is not a valid model directory.
        """
        assert os.path.exists(path)
        config_file = os.path.join(path, cls.model_files[0])
        return cls(config_file)

    def _load_config(self, config_file):
        config = get_config(config_file, STATIC_VIDEO_CFG_SPEC.splitlines())
        for key, values in config.iteritems():
            for name, value in values.items():
                self.__dict__["{}_{}".format(key, name)] = value

        self.target_labels = {self.PHOTO: self.classifier_photo_tl_id,
                              self.SLIDESHOW: self.classifier_slideshow_tl_id}

    def classify_video(self, video_descriptors):
        """ Gets class of video given optical flow descriptors """
        if video_descriptors:
            video_descriptor = video_descriptors[0]
            static_flow_score = video_descriptor[3]
            total_flow = np.mean(video_descriptor[1])
            if static_flow_score > self.classifier_video_th:
                if total_flow > self.classifier_slideshow_th:
                    return self.SLIDESHOW
                return self.PHOTO
            return self.VIDEO
        return self.UNKNOWN

    def compute_descriptors(self, video_obj):
        """
        Computes the descriptors for the input video.
        Empty list returned if there are not enough frames.
        """
        v_name = video_obj.video_name_path
        v_length = video_obj.video_length
        possible_offsets = np.array([v_length / 2, 60])
        off = np.min(possible_offsets[possible_offsets >= 0])
        logger.info('\t 1.1. Sampling video %s (offset %.2f)'
                    % (video_obj.video_name_path, off))

        sampler = VideoFFmpegSampler(
            v_name, duration=self.sampling_duration, offset=off,
            fps=self.sampling_fps, scale=self.sampling_scale)
        list_of_frames = sampler.sample(
            output_dir=video_obj.frames_folder_path)

        logger.info('\t 1.2. Computing descriptors')
        if len(list_of_frames) >= self.MIN_FRAMES:
            return self.optical_flow_extractor.compute_dense_opticalflow(list_of_frames)
        return []

    def create_test_flow(self):
        flow = Flow("Static_Video")
        feature_extraction = Step("feature_extraction", self,
                                  "compute_descriptors")
        classify = Step("classify", self, "classify_video")
        for step in [feature_extraction, classify]:
            flow.add_step(step)
        flow.connect(feature_extraction, classify, feature_extraction.output)
        flow.start_with(feature_extraction, FutureFlowInput(flow, 'video_obj'))
        flow.output = classify.output
        return flow
