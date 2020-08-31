import os
import shutil
import numpy as np

from scipy.spatial.distance import euclidean, cosine
from logging import getLogger
from affine.video_processing import run_cmd
from affine.detection.vision.utils.gist import Gist

__all__ = ['VideoFFmpegSampler']
logger = getLogger(__name__)


class VideoFFmpegSampler(object):

    """
    Class to sample a video using FFmpeg, including different extraction
    and filtering options for the returned sampled frame list
    """

    def __init__(self, filename, duration=30, offset=0, fps=1.0, scale=1,
                 key_frames=False):
        """
        Initialize a VideoFFmpegSampler with the given configuration

        Args:
            filename: string, path to the video file
            duration: int, number of seconds that we want to sample from the video
            offset: int, number seconds that we want to skip at the beginning\
                of the video (i.e., when do we want to start sampling).
                If offset >= 0: sampling will skip that amount of seconds before start sampling
            fps: float, frames per second that we want to sample
            scale:  If >0 and <=1, it is the "ratio" to resize the frames
                    If > 1, it is the desired width for the sampled frames\
                        (ffmpeg will calculate the height to avoid deformation)
            key_frames: bool, to activate the option to extract the I-frames from ffmpeg.
                NOTE: it is NOT compatible with a specifc fps. If it is TRUE, fps is ignored.
        Returns:

        Raises/Assertions:
            Asserts if the input video file does not exist
                or if the params have invalid values
        """
        assert os.path.exists(filename), "This video file doesn't exist"
        assert duration > 0, \
            "Sampling duration has to be greater than 0 seconds"
        assert fps >= 0.1, \
            "Sampling rate has to be >= than 0.1 frames per second"
        assert scale > 0, \
            "Scale has to be greater than 0"

        self.filename = filename
        self.duration = duration
        self.offset = offset
        self.fps = fps
        self.scale = scale
        self.keyframes = key_frames

        self.GIST_D_MAX = 0.8
        self.GIST_D_MIN = 0.2
        self.GIST_RATIO = 1.5
        self.img_ext = 'jpg'
        self.distance = euclidean

    def sample(self, output_dir):
        """
        Sample video loaded in the VideoFFmpegSampler object using ffmpeg.

        Args:
            output_dir: string with path to folder where we want to store sampled frames.
            NOTE!! If output_dir folder already exists:
            THE WHOLE FOLDER WILL BE DELETED and the frames extracted in a clean new folder

        Returns:
            list of strings with each of the sampled frames (full path filenames)
        Raises/Assertions:
        """
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # By specifying -ss (offset) after -i, we sample from the beginning and
        # and drop the first frames. This way we can still sample corrupted
        # videos where it's not possible to first seek to the offset.
        list_of_ffmpeg_params = ['ffmpeg', '-i', self.filename,
                                 '-ss', str(self.offset), '-t',
                                 str(self.duration)]

        filter_strings = ''
        if self.scale > 1:
            filter_strings = 'scale=%d:-1' % (self.scale)
        elif self.scale < 1:
            filter_strings = 'scale=iw*%f:-1' % (self.scale)

        if not self.keyframes:
            list_of_ffmpeg_params += ['-r', str(self.fps)]
        else:
            list_of_ffmpeg_params += ['-vsync', '0']
            if len(filter_strings) > 0:
                filter_strings = str(filter_strings) + ', '
            filter_strings += 'select=eq(pict_type\,I)'
        if len(filter_strings) > 0:
            list_of_ffmpeg_params += ['-vf', filter_strings]

        list_of_ffmpeg_params += ['-f', 'image2',
                                  output_dir + '/%06d.' + self.img_ext]
        run_cmd(list_of_ffmpeg_params)

        list_of_frames = os.listdir(output_dir)
        list_of_frames.sort()
        list_of_filenames = \
            [os.path.join(output_dir, f)
             for f in list_of_frames if f.endswith(self.img_ext)]

        return list_of_filenames

    def _compute_features(self, list_imgs):
        """
        Computes gist features for all images in the list.
        Gist extractor resizes and crops the center squared patch if necessary
        for all images to 128x128.

        Args:
            list_imgs: list of full path image file names
        Returns:
            2-d matrix, with one gist feature in each row (one per input image)
        """
        gist_params = {'orientations': [4, 4, 4, 4], 'nblocks': 4}
        gist = Gist(**gist_params)
        gist.createGabors()

        all_desc_gist = np.zeros((len(list_imgs), 256))
        for i, img_path in enumerate(list_imgs):
            desc_gist = gist.compute_gist_descriptor(img_path)
            all_desc_gist[i, :] = desc_gist

        return all_desc_gist

    def _filter_online(self, desc_matrix, list_imgs):
        """
        This method first gets all distances between each frame
        and its previous one (according to the descriptors given for each frame)
        and then returns a sub list of the input image list with those that show
        the biggest appearance changes.

        Args:
            desc_matrix: 2D numpy array with 1 image descriptor per row
            list_imgs: list with at least one image file
        Returns:
            filtered_frames: filtered subset of input list_imgs
            filtered_frames_indx: list of indexes in list_imgs of the filtered frames
            norm_d_list: list of normalized distance of each input img with its previous one
        Asserts:
            number of rows in desc_matrix has to be equal to the number of images in the input list
        """
        assert (len(list_imgs) == desc_matrix.shape[0]), \
            "desc_matrix needs to have one row for each image in the list"

        def _filter_nan(d):
            """ if d is Nan return 0 """
            new_d = d
            if np.isnan(d):
                new_d = 0
            return new_d

        first_diff = 0
        diff_list = [self.distance(desc_matrix[i - 1], desc_matrix[i])
                     for i in range(1, len(list_imgs))]
        diff_list = np.array([first_diff] + [_filter_nan(d)
                                             for d in diff_list])
        max_diff = np.max(diff_list)
        if max_diff:
            norm_d_list = np.array(diff_list / max_diff)
        else:
            norm_d_list = np.array(diff_list)

        gist_d_accept_th = np.max([self.GIST_D_MAX, np.mean(norm_d_list)])
        filtered_frames_indx = [0]
        for fr_i in range(1, len(list_imgs)):
            if (norm_d_list[fr_i] > self.GIST_RATIO * norm_d_list[fr_i - 1]
                    and norm_d_list[fr_i] > self.GIST_D_MIN) \
                    or norm_d_list[fr_i] >= gist_d_accept_th:
                filtered_frames_indx += [fr_i]

        filtered_frames = [list_imgs[i] for i in filtered_frames_indx]

        return filtered_frames, filtered_frames_indx, norm_d_list

    def filter_frames(self, list_imgs):
        """
        Given a list of frames, select those more distinct from the rest.
        This difference is computed using the gist descriptor.
        A frame is stored if its difference with previous frame is more
            than twice the difference of previous pair of frames
            OR above a th distance value normalized for this sequence
        First frame is always stored as it's an special case.

        Args:
            list_imgs: list of image files
        Returns:
            filtered_frames: filtered subset of input list_imgs
            filtered_frames_indx: list of indexes in list_imgs of the filtered frames
            norm_d_list: list of normalized distance of each input img with its previous one

        """
        filtered_frames = filtered_frames_indx = norm_d_list = []
        if list_imgs:
            desc_matrix = self._compute_features(list_imgs)

            filtered_frames, filtered_frames_indx, norm_d_list = \
                self._filter_online(desc_matrix, list_imgs)

        return filtered_frames, filtered_frames_indx, norm_d_list
