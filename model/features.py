import pickle
import math
import os
import shutil
import struct
import ner
import numpy as np
import cv2
import scipy


from abc import ABCMeta, abstractmethod
from collections import defaultdict
from datetime import datetime
from logging import getLogger
from multiprocessing import cpu_count
from socket import error as socket_error

from affine.detection.nlp.nec.detection import _preprocess_text
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge
from tempfile import mkdtemp, mkstemp
from xml.sax.saxutils import XMLGenerator
from scipy.cluster.vq import vq

import affine.detection.utils.bounding_box_utils as bbu
from affine.model import *
from affine import config
from affine.detection.model.imagedesc import ImageDesc
from affine.detection.nlp.ner.ner_builder import EntityExtractor, NER_PORT
from affine.detection.nlp.keywords.keyword_matching import PageEntityMatcher, \
    process_text
from affine.detection.model.nms import edge_nms, box_nms
from affine.detection.model.structured_edge_detection import \
    structured_edge_detection
from affine.detection.model_worker.utils import merge_list, batch_list
from affine.video_processing import run_cmd
from affine.parallel import pool_map
from util import PickleMixin

__all__ = ['AbstractFeature', 'RawPatch', 'NerFeatureExtractor',
           'ColorHistFeatureExtractor', 'OpticalFlowFeatureExtractor',
           'LocalFeatureExtractor', 'LocalFeatureDetector', 'SurfExtractor',
           'SpatialBagOfWords', 'BagOfWords', 'SlidingWindowBoxExtractor',
           'EdgeBoxExtractor', 'GridBoxExtractor', 'BoundingBoxRegressor']

logger = getLogger('affine.detection.model.features')


class AbstractFeature(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def extract(self, face_ids):
        """extract features for given box_ids"""

    def save_to_file(self, file_path):
        raise NotImplementedError()

    @classmethod
    def load_from_file(cls, file_path):
        raise NotImplementedError()


def extract_for_chunk(face_ids):
    from affine.detection.vision.facerec.face_sig_extract import FaceSigExtractor
    feats = [None] * len(face_ids)
    try:
        patches = []
        index_map = {}
        temp_dir = mkdtemp()
        for i, face_id in enumerate(face_ids):
            box = Box.get(face_id)
            assert box, "Box with id %s missing" % face_id
            path = os.path.join(temp_dir, str(face_id))
            try:
                img_path = box.download_image(path)
                rect = (box.x, box.y, box.width, box.height)
                patches.append((img_path, rect))
                index_map[i] = len(patches)-1
            except Exception:
                pass

        logger.info('Found %d/%d boxes on S3' % (len(patches), len(face_ids)))

        if patches:
            fse = FaceSigExtractor()
            signatures = fse.extract_signatures(patches)

            feats = []
            for i in xrange(len(face_ids)):
                if i in index_map:
                    feats.append(signatures[index_map[i]])
                else:
                    feats.append(None)

        shutil.rmtree(temp_dir)
        return feats
    except Exception:
        box_str = ','.join(map(str, face_ids))
        logger.error(
            "Failed to extract for the chunk of box_ids : %s" % box_str)
        return feats


class RawPatch(AbstractFeature):
    MAX_FACES_EXTRACT = 50

    def extract(self, box_ids):
        ''' returns a numpy array where each row is the feature of the \
            corresponding box_id.
            Note: Some rows can be None, if we failed to extract signature \
            for some reason (eg: image not available on S3)
            In such cases, the returned numpy array is not an iterable object \
            (feats.shape = ())
            So best way to use the features is to convert feats to a list of \
            lists --> feats.tolist()
            And then check for each row if row != None
        '''
        if box_ids and not isinstance(box_ids, list):
            box_ids = [box_ids]

        t1 = datetime.utcnow()
        logger.info('Extracting signatures for %d boxes' % len(box_ids))
        chunks = []
        for i in range(0, len(box_ids), self.MAX_FACES_EXTRACT):
            chunks.append(box_ids[i:i + self.MAX_FACES_EXTRACT])
        num_proc = min(cpu_count(), len(chunks))
        features_list = pool_map(extract_for_chunk, chunks, processes=num_proc)
        feats = []
        for f in features_list:
            feats += f
        total_time = (datetime.utcnow() - t1).seconds
        logger.info('Completed Extraction for %d boxes in %s seconds' %
                    (len(box_ids), total_time))
        return np.asarray(feats)


class NerFeatureExtractor(AbstractFeature):
    N_SPLITS = 3

    def __init__(self):
        self.ee = EntityExtractor()
        self.tagger = ner.SocketNER(host='localhost', port=NER_PORT)

    def get_entities(self, text):
        try:
            return self.tagger.get_entities(text)
        except socket_error, err:
            self.ee.start_ner_server()
            return self.tagger.get_entities(text)

    def extract(self, page_id):
        logger.info('Extracting NER features for page: %d' % page_id)
        page = WebPage.get(page_id)
        text = _preprocess_text(page.title_and_text)
        entity_label_ids = self.get_entity_label_ids(text)
        ftr_dict = {}
        for label_id in entity_label_ids:
            ftr_dict[label_id] = self.entity_featurize(page, label_id)
        return ftr_dict

    def get_entity_label_ids(self, text):
        """ Runs NER on given text, looks up extracted entities in the index
            and returns associated labels
        Args:
            text: Page text for entity extraction
        Returns:
            Label ids from index for matched entities
        Examples:
            >> nfe.get_entity_label_ids('Kobe Bryant plays for LA Lakers')
            [<Basketball_id>]
        """
        ner_dict = self.get_entities(text)
        persons = set(ner_dict.get('PERSON', []))
        orgs = set(ner_dict.get('ORGANIZATION', []))
        candidate_label_ids = set()
        for entity_list, entity_type in \
                (persons, 'person'), (orgs, 'organization'):
            if entity_list:
                candidate_label_ids.update([i for (i,) in session.query(
                    NamedEntity.label_id.distinct()).
                    filter(NamedEntity.name.in_(entity_list),
                           NamedEntity.entity_type == entity_type)]
                )
        return candidate_label_ids

    def entity_featurize(self, page, label_id):
        """Uses indexed named entities for a label to extract
        feature values from page text
        Args:
            page: page to extract features from
            label_id: label to extract features for
        Returns:
            Numpy feature array
        Examples:
            >> nfe.entity_featurize(WebPage.get(98765), 1234)
            array([3, 2, 1, 2, 1])
        """
        pem = PageEntityMatcher()
        # Might not be a good idea to create the structure for every page
        for ne in NamedEntity.query.filter_by(label_id=label_id):
            pem.add_entity(ne.id, ne.name)
        # one value per split, uniques, in_title
        fv = np.zeros(self.N_SPLITS + 2, dtype=int)
        processed_body = process_text(
            page.description_text or '', stemming=False)
        processed_title = process_text(page.title or '', stemming=False)
        all_matches = set()
        for i, sp in enumerate(self._split_list(processed_body, self.N_SPLITS)
                               + [processed_title]):
            matches = pem.matching_entities(sp)
            fv[i] = len(matches)
            all_matches.update(matches)
        fv[i + 1] = len(all_matches)
        return fv

    def _split_list(self, tt, n_splits):
        """ Simple function to split a list"""
        assert n_splits > 0
        sz = max(1, len(tt) / n_splits)
        return [tt[i: i + sz] for i in range(0, (n_splits - 1) * sz, sz)]\
            + [tt[(n_splits - 1) * sz:]]


# IMAGE FEATURES
class ColorHistFeatureExtractor(AbstractFeature):

    """
    This feature extractor will compute a color histogram based \
    descriptor for each image given to the extractor
    """

    def __init__(
        self, color_spaces_used=['RGB', 'Lab', 'HSV'], numBins=16, norm=True
    ):
        """
        Config. and initialize the color histogram feature extractor

        NOTE that in opencv the different color spaces are handled with the\
        following ranges (which are the ones assumed in this class):
        RGB: R [0,255], G [0,255], B [0,255]  (Actually openCV loads BGR)
        HSV: Hue range:[0,179], Saturation range:[0,255], Value range:[0,255]
        Lab:  0 <= L <= 100 , -127 <= a <= 127, -127 <= b <= 127
        BUT cv2.CVT already maps each of this channels to [0,255]

        Args:
            color_spaces_used: list of strings that describe the \
                color spaces we want to consider to build the descriptor.
                It could be any combination of these three: 'RGB' 'Lab' 'HSV'
                e.g., ['RGB', 'Lab', 'HSV'] or ['RGB', 'Lab']
                Only these 3 colorspaces are supported.
            numBins: int, number of bins for the histogram of EACH color channel
                Default value is 16.
            norm: compute normalized histograms (True by default).
                If False the histograms provide "raw" pixel count of each color

        Returns:

        Raises/Assertions:

        """
        numcolorspaces = len(color_spaces_used)
        assert (numcolorspaces > 0 and numcolorspaces < 4), \
            "Invalid list of colorspaces"
        assert (numBins >= 8 and numBins <= 256), "Invalid number of bins"

        self.colorspaces = color_spaces_used
        self.numChannels = 3
        self.numBins = numBins
        self.norm = norm
        self.minValues = {}
        self.numberBins = {}
        self.maxValues = {}
        for c in self.colorspaces:
            self.minValues[c] = [0] * self.numChannels
            self.numberBins[c] = [numBins] * self.numChannels
            if c == 'HSV':
                self.maxValues[c] = [179, 255, 255]
            else:
                self.maxValues[c] = [255] * self.numChannels

    def get_desc_length(self):
        """
        return the length of the descriptors computed with this feature extractor
        """
        return self.numBins * self.numChannels * len(self.colorspaces)

    def extract(self, list_of_imagefiles):
        """
        Computes a color descriptor for each image in list_of_imagefiles

        This feature contains concatenated histograms according to the color \
        spaces used in this extractor (self.colorspaces)

        Args:
            list_of_imagefiles: list of images to extract color features from
        Returns:
            numpy array where each row is the feature of the corresponding image

            NOTE ZERO ROW meaning:
                Some rows can be all zeros if the descriptor extraction was \
                wrong or failed for any reason (e.g., gray scale image,
                image not accesible, corrupted file or any other issue)

        Raises/Assertions:
            This function assumes that all the given imagefile names exist,
            otherwise, it asserts "Image does not exist"

        """
        nImgs = len(list_of_imagefiles)
        descriptorLength = self.get_desc_length()
        colorDescriptors = np.zeros((nImgs, descriptorLength))

        for imfile, i in zip(list_of_imagefiles, range(nImgs)):

            img_bgr = cv2.imread(imfile)
            if img_bgr is not None and img_bgr.shape[2] == 3:
                im_desc = []
                for c in self.colorspaces:
                    im_desc_tmp = self.get_color_descriptors(
                        img_bgr, colorspace=c, normalize=self.norm)
                    im_desc = np.append(im_desc, im_desc_tmp)

                if len(im_desc) == descriptorLength:
                    colorDescriptors[i, :] = im_desc
            else:
                logger.error(
                    'Image %s should be a valid color image file' % imfile)
                assert os.path.exists(imfile), "Image does not exist"

        return colorDescriptors

    def get_color_descriptors(self, img_bgr, colorspace,
                              normalize=True, minVal=0, maxVal=255, numBins=256
                              ):
        """
        Compute color descriptors for image img_bgr.

        Args:
            img_bgr: COLOR image previously loaded with cv.imread.
            colorspace: string that indicates the color space to be used
                'RGB', 'Lab' or 'HSV'

            normalize: boolean, normalize the resulting histogram.
                True by default
            minVal, maxVal, numBins: parameters from openCv calcHist function,
                used to define the range of values considered and the number of\
                 bins of the resulting histogram

        Returns:
            numpy array with concatenated histograms from each channel from \
            the required colorspace for the input image

        Raises/Assertions:
            It assumes that the input image img_bgr has 3 channels.
                Asserts otherwise.
            Asserts if the parameters numBins or maxVal are invalid
        """
        assert (numBins > 0), "Number of bins in the histogram should be > 0"
        assert (maxVal >= minVal), \
            "maxVal of histogram ranges needs to be greater than the minVal"

        colorChannels = self._get_colorchannels(img_bgr, colorspace)
        minValues = self.minValues[colorspace]
        maxValues = self.maxValues[colorspace]
        numberBins = self.numberBins[colorspace]
        im_desc = []
        for band, ch in zip(colorChannels, range(self.numChannels)):
            if len(band) > 0:
                hist_band = cv2.calcHist(
                    [band], [0], None, [int(numberBins[ch])],
                    [minValues[ch], maxValues[ch] + 1])
                if normalize:
                    numPix = np.sum(hist_band)
                    hist_band = hist_band / numPix
                else:
                    hist_band = np.int32(np.around(hist_band))

                im_desc = np.append(im_desc, hist_band)

        return im_desc

    def _get_colorchannels(self, img_bgr, colorspace='RGB'):
        """
        Split the image into the required colorspace channels

        Args:
            img_bgr: COLOR image previously loaded with cv.imread.
            colorspace: string that indicates the color space to be used
                'RGB', 'Lab' or 'HSV'
                Default case is RGB

        Returns:
            list with the split image channels according to the given colorspace
            For example, if colorspace is RGB, this will return a list with the
            blue, red and green channels

        Raises/Assertions:

        """
        if colorspace == 'HSV':
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(img_hsv)
            colorChannels = [H, S, V]
        elif colorspace == 'Lab':
            img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            L, a, b = cv2.split(img_lab)
            colorChannels = [L, a, b]
        else:
            blue, green, red = cv2.split(img_bgr)
            colorChannels = [blue, green, red]

        return colorChannels


class OpticalFlowFeatureExtractor(AbstractFeature):

    """ Extractor of optical flow based video features.
    This class can provide both per frame or per video descriptors.
    """

    BASIC_FLOW, MASKED_FLOW, STATIC_FLOW = [0, 1, 2]

    def __init__(
        self, get_static_flow=False,
        get_masked_flow=False, numBins=8,
        flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5,
        poly_sigma=1.2, flags=0
    ):
        """
        Initialize the optical flow descriptor extractor with the given params.

        Args:
            get_masked_flow: boolean that indicates if we want a masked version
                of the flow descriptor.
                IF FALSE: computes the flow descriptors in the whole frame.
                IF TRUE: computes the flow descriptors in 3 different image regions
                    (whole image, inner part, the outer part):
                    For the outter part, we consider a border around the image of
                    width/4 pixels for top/bottom, and height/4 pixels for left/right borders
                    Inner part is the rest of the image, without this outter frame.
            numBins = number of bins for each of the optical flow histograms
                (we compute a histogram with the distribution of dominant_flow_orientation
                    and other with total_flow)
            flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags:
                standard Opencv cv2.calcOpticalFlowFarneback parameters

        Returns:

        Raises/Assertions:

        """
        assert numBins > 0, "Number of bins for each histogram has to be > 0"

        self.flow_type = self.BASIC_FLOW
        assert not (get_masked_flow and get_static_flow), \
            'Cannot have masked and static flow results together'
        if get_masked_flow:
            self.flow_type = self.MASKED_FLOW
        elif get_static_flow:
            self.flow_type = self.STATIC_FLOW
        self.numBins = numBins

        self.flow = flow
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags

        self.rangeColsOut = None
        self.rangeRowsOut = None
        self.rangeRowsIn = None
        self.rangeColsIn = None

        self.width = None
        self.height = None

    def get_desc_length(self):
        """
        return the length of the descriptors computed with this feature extractor
        """
        desc_length = self.numBins * 2
        if self.flow_type == self.MASKED_FLOW:
            desc_length = desc_length * 3

        return desc_length

    def extract_multiple(self, list_sorted_frames):
        features = [self.extract(sorted_frames) for sorted_frames in
                    list_sorted_frames]
        return np.vstack(features)

    def extract(self, sorted_frames):
        """
        Extract the optical flow based descriptor for the set of given frames.

        Args:
            sorted_frames: sorted list of full path! frame/image file names.
                           ASSUMES their sorted names correspond with their
                           order in the video sequence

        Returns:
            Two histograms concatenated: [ ]
            NOTE: If the given frame list was not suitable to compute this
            descriptor, the returning matrix will be ALL ZEROS

        """
        assert (sorted_frames and len(sorted_frames) > 0), \
            "List of frames can't be empty"

        list_sorted_f = np.array(sorted_frames)

        if len(list_sorted_f) >= 2:
            o_flow = self.compute_dense_opticalflow(
                sorted_frames=list_sorted_f)
            oflow_descriptors = self.build_opticalflow_desc(o_flow)
            oflow_descriptors = np.array(oflow_descriptors, ndmin=2)
        else:
            oflow_descriptors = np.zeros((1, self.get_desc_length()))

        return oflow_descriptors

    def _init_mask(self, img):
        """
        Initialize the range of rows and columns needed for the masked version
        of the optical flow for images like the input img

        Args:
            img: image previously loaded with cv2.imread

        Returns:

        Raises/Assertions:
        """

        height = img.shape[0]
        width = img.shape[1]

        f_col = width / 4
        l_col = f_col * 3
        f_row = height / 4
        l_row = f_row * 3
        self.rangeColsOut = range(0, f_col) + range(l_col, width)
        self.rangeRowsOut = range(0, f_row) + range(l_row, height)
        self.rangeRowsIn = range(f_row, l_row)
        self.rangeColsIn = range(f_col, l_col)

        self.width = width
        self.height = height

    def _get_maskedflow(self, flow_all):
        """
        Given the optical flow result for each pixel in an image (flow_all),
        mask the values corresponding to the outer image frame and the inner
        image frame

        Args:
            flow_all: matrix with optical flow values computed for all pixels in an image
                NOTE that flow_all should be the output of a function that computes
                dense optical flow values, i.e., for all pixels, for example:
                    cv2.calcOpticalFlowFarneback

        Returns:
            two matrices (inner_flow, outer_flow) corresponding to the input
            optical flow values (flow_all) that correspond to the inner or outer
            part of the image respectively

        Raises/Assertions:
        """

        if (self.rangeRowsOut == None) or (self.rangeRowsIn == None) \
                or (self.rangeColsOut == None) or (self.rangeColsIn == None) \
                or (self.height == None) or (self.width == None):
            self._init_mask(np.array(flow_all))

        assert (flow_all.shape[0] == self.height
                and flow_all.shape[1] == self.width), 'Wrong flow matrix size'

        outer_flow = flow_all[self.rangeRowsOut, :, :]
        flow_all_mid = flow_all[self.rangeRowsIn, :, :]

        outer_flow2 = flow_all_mid[:, self.rangeColsOut, :]
        inner_flow = np.array(flow_all_mid[:, self.rangeColsIn, :])

        if outer_flow.shape[0] > outer_flow2.shape[0]:
            off = outer_flow.shape[0] - outer_flow2.shape[0]
            outer_flow = np.append(outer_flow[:-off, :], outer_flow2, axis=1)

        elif outer_flow.shape[0] < outer_flow2.shape[0]:
            off = outer_flow2.shape[0] - outer_flow.shape[0]
            outer_flow = np.append(outer_flow, outer_flow2[:-off, :], axis=1)

        else:
            outer_flow = np.append(outer_flow, outer_flow2, axis=1)

        return inner_flow, outer_flow

    def compute_dense_opticalflow(self, sorted_frames):
        """
        Compute dense optical flow using all frames in sorted_frames.
        IF self.mask is True, it computes inner and outer frame flow \
            besides the standard one (whole image)
        Args:
            sorted_frames: frames to use for the optical flow computation.
                It is assumed to be sorted.
        Returns: returns the flow results in 3 histograms,
            each of them with max magnitude, dominant orientation and total flow
            between every 2 consecutive frames).
            If flow_type is STATIC_FLOW then we have a 4th value which is the
            static_flow_score

        Raises/Assertions:
            NOTE: this function ASSUMES that the length of sorted_frames is > 2
            Otherwise, it asserts.
        """
        assert (len(sorted_frames) > 1),\
            "Optical flow can't be computed with less than 2 frames"

        frames = np.array(sorted_frames)
        img = cv2.imread(frames[0])
        prvs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pix_orient = []
        total_flow = []
        max_mag = []
        perc_list = []
        self.area = float(img.shape[0] * img.shape[1])

        if self.flow_type == self.MASKED_FLOW:
            self._init_mask(img)
            pix_orient_in = []
            total_flow_in = []
            max_mag_in = []
            pix_orient_out = []
            total_flow_out = []
            max_mag_out = []

        static_flow_score = 0
        # % of pixels which are moving in the frame
        perc_prev = 0

        for fr in frames[1:]:
            img = cv2.imread(fr)
            nxt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            flow_all = cv2.calcOpticalFlowFarneback(
                prvs, nxt, flow=self.flow, pyr_scale=self.pyr_scale,
                levels=self.levels, winsize=self.winsize,
                iterations=self.iterations, poly_n=self.poly_n,
                poly_sigma=self.poly_sigma, flags=self.flags
            )

            if self.flow_type == self.MASKED_FLOW:
                inner_flow, outer_flow = self._get_maskedflow(flow_all)

            self._get_flow_stats(flow_all, pix_orient, total_flow, max_mag,
                                 perc_list)
            if perc_list[-1] == 0 and perc_prev == 0:
                static_flow_score += 1
            perc_prev = perc_list[-1]
            if self.flow_type == self.MASKED_FLOW:
                self._get_flow_stats(
                    outer_flow, pix_orient_out, total_flow_out,
                    max_mag_out)
                self._get_flow_stats(
                    inner_flow, pix_orient_in, total_flow_in, max_mag_in)

            prvs = nxt

        if self.flow_type == self.STATIC_FLOW:
            res = [[pix_orient, total_flow, max_mag,
                    static_flow_score/float(len(perc_list))]]
        else:
            res = [[pix_orient, total_flow, max_mag]]
            if self.flow_type == self.MASKED_FLOW:
                res = [res[0],
                       [pix_orient_out, total_flow_out, max_mag_out],
                       [pix_orient_in, total_flow_in, max_mag_in]]
        return res

    def _get_flow_stats(self, flow, pix_orient, total_flow, max_mag,
                        perc_list=None):
        """
        Computes several statistics of the values in magnitude and orientation
        for the optical flow computed in matrix flow
        and APPENDS these statistics in the given arrays: pix_orient, total_flow, max_mag

        Args:
            flow: matrix with optical flow values computed for a set of pixels
                computed with a dense optical flow function, for example:
                    cv2.calcOpticalFlowFarneback

            pix_orient, total_flow, max_mag: arrays where we want to APPEND the
                following statistics of the input flow matrix:
                    - dominant pixel orientation
                    - total amount of flow, i.e., how many pixels have a flow magnitude > 0
                    - maximum optical flow magnitude

                NOTE these three input arrays are modified inside this function

        Returns:

        Raises/Assertions:

        """
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        nang = ang * 180 / np.pi / 2
        if perc_list is not None:
            row = np.nonzero(mag > 1)
            perc = len(row[0]) / self.area
            perc = np.around(perc, decimals=2)
            perc_list.append(perc)

        mag_hist, _ = np.histogram(mag, bins=range(0, 256))
        mag_hist = mag_hist / float(flow.shape[0] * flow.shape[1])

        nang_hist, _ = np.histogram(nang, bins=range(0, 181))
        nang_hist = nang_hist / float(flow.shape[0] * flow.shape[1])

        pix_orient.append(np.around(max(nang_hist), decimals=2))
        total_flow.append(np.around(1 - mag_hist[0], decimals=2))
        max_mag.append(np.max(mag))

    def build_opticalflow_desc(self, opt_flow_stats):
        """
        Given the optical flow statistics obtained for a set of input frames
        using self.compute_dense_opticalflow, this method computes
        a single compact descriptor for the whole set of frames.
        It computes a histogram of the total amount of flow
        and dominant flow orientation values across all the set of frames.

        Args:
            opt_flow_stats: numpy array that contains in each component a set of three lists.
                Each of these lists contains dominant pixel orientation,
                total flow and maximum magnitude values,
                respectively, for each of the frames considered,
                as obtained when calling self.compute_dense_opticalflow

        Returns:
            Returns a list of histograms.
            1-dimensional numpy array that contains a histogram of the
                dominant pixel orientations (self.numBins and range [0-360])
                and a histogram of the total amount of flow (self.numBins and range [0-1000])
                across all frames
                Length is 1 if its not masked and 3 with the mask [full, outer,
                inner]

        Raises/Assertions:

        """

        current_video_desc = []
        for row in opt_flow_stats:
            pix_orient = row[0]
            total_flow = row[1]
            numSamples = len(pix_orient)
            img = np.array(pix_orient, dtype=np.float32)
            hist_or = cv2.calcHist(
                [img], channels=[0], mask=None, histSize=[self.numBins],
                ranges=[0, 1])
            hist_or = hist_or / float(numSamples)

            img = np.array(total_flow, dtype=np.float32)
            hist_tot = cv2.calcHist(
                [img], channels=[0], mask=None, histSize=[self.numBins],
                ranges=[0, 1])
            hist_tot = hist_tot / float(numSamples)

            current_desc = np.append(hist_or, hist_tot)
            current_video_desc = np.append(current_video_desc,
                                           current_desc)

        return current_video_desc


class LocalFeatureExtractor(PickleMixin):

    def __init__(self, feature_type, keyp_grid_params=None):
        """
        Initialize feature extractor.
            If keyp_grid_params is empty, the extractor computes
            both features location (keypoints) and descriptor.
            Otherwise, it uses the params given to compute a fixed-grid of keypoint locations

        Args:
            feature_type: string (it has to be 'SURF' or 'BRISK')
            keyp_grid_params = [fix_im_width, fix_im_height, grid_interval]
                parameters for the grid of fixed keypoint locations
                if it's empty, locations and automatically detected by the algorithm
                e.g. keyp_grid_params = [320, 240, 18]
                Resize images to be (240 rows x 320 columns) and create a list of keypoint locations
                equally spaced in the image with an interval of 18 pixels.

        Returns:

        Raises/Assertions:
            Asserts if the feature type is not 'SURF' or 'BRISK'
            Asserts if the keyp_grid_params has a wrong number of elements
                (it should be empty or have 3 positive numbers)

        """
        assert feature_type in ['SURF', 'BRISK'], \
            "Feature type not supported"

        assert not keyp_grid_params \
            or (len(keyp_grid_params) == 3 and np.min(keyp_grid_params) > 0), \
            "Keypoint_grid_params list should be empty or contain 3 positive numbers"

        self._setup(feature_type, keyp_grid_params)

    def _setup(self, feature_type, keyp_grid_params):
        # Setting vals for save_to_file
        self._keyp_grid_params = keyp_grid_params
        self._feature_type = feature_type
        self._setup_extractor()
        self._setup_detector()

    def _setup_extractor(self):
        self.extractor = cv2.DescriptorExtractor_create(self._feature_type)

    def _setup_detector(self):
        self.detect_keypoints = True

        if self._keyp_grid_params and len(self._keyp_grid_params) == 3 \
                and np.min(self._keyp_grid_params) > 0:

            self.FIX_IM_WIDTH = self._keyp_grid_params[0]
            self.FIX_IM_HEIGHT = self._keyp_grid_params[1]
            self.detect_keypoints = False

        self.keypoint_detector = LocalFeatureDetector(
            self._feature_type, self._keyp_grid_params)

    def extract(self, list_of_imagefiles):
        """
        Computes and returns the keypoints and descriptors for each image
        in a list of tuples.
        This method assumes that the given image exist

        Args:
            list_of_imagefiles: list with full path image file names
                where we want to compute the features

        Returns: a list of tuples containing keypoints and descriptors for each
            image in the list.
            Each tuple consists of: (list_of_cv2.KeyPoints, array_with_descriptors)
            Each row in the array contains the descriptor of the corresponding KeyPoint

        Assertions/Exception:
            Function compute_features will assert if image does not exist

        """

        all_features = []

        for im_name in list_of_imagefiles:
            feat = self.compute_features(im_name)
            all_features = all_features + [feat]

        return all_features

    def compute_features(self, im_name):
        """
        Computes local features on the given image file.

        Args:
            im_name: full path image file name where we want to compute the features
        Returns:
            Tuple with the keypoints and descriptors found in the input image.
            This tuple consists of: (list_of_cv2.KeyPoints, array_with_descriptors)
            Each row in the array contains the descriptor of the corresponding KeyPoint

        Assertions/Exception:
            Asserts if the image does not exist
        """
        assert os.path.exists(im_name), "Image %s does not exist" % (im_name)

        img_gray = cv2.imread(im_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        if not self.keypoint_detector.detect_keypoints:
            img_gray = cv2.resize(img_gray,
                                  (self.keypoint_detector.FIX_IM_WIDTH,
                                   self.keypoint_detector.FIX_IM_HEIGHT))

        img_keypoints = self.keypoint_detector.get_keypoints(img_gray)

        img_keypoints, img_descriptors = self.extractor.compute(img_gray,
                                                                img_keypoints)

        return img_keypoints, img_descriptors

    def __getstate__(self):
        return {
            'feature_type': self._feature_type,
            'keyp_grid_params': self._keyp_grid_params,
        }

    def __setstate__(self, state):
        ft, grid_params = state['feature_type'], state['keyp_grid_params']
        self._setup(ft, grid_params)


class SurfExtractor(LocalFeatureExtractor):

    def __init__(self, upright=True, extended=True, hessian_thresh=400.00,
                 keyp_grid_params=None, keypoint_limit=None):
        """ Initializes a SurfExtractor which is a child of LocalFeatureExtractor
        Args:
            upright: if False, means that detector computes orientation of each feature.
            otherwise the orientation is not computed (which is much faster)

            extended: if True the extended descriptors (128 elements each) would get computed

            hessian_thresh: Threshold for the keypoint detectotor
            higher threshold results in less keypoints

            keyp_grid_params = [fix_im_width, fix_im_height, grid_interval]

            keypoint_limit: limit for the number of keypoints returned from the extract function

        """
        super(SurfExtractor, self).__init__(
            'SURF', keyp_grid_params=keyp_grid_params)
        self.upright = upright
        self.extended = extended
        self.hessian_thresh = hessian_thresh
        self.keypoint_limit = keypoint_limit

    @property
    def upright(self):
        """ returns the upright value"""
        return self._upright

    @upright.setter
    def upright(self, upright):
        """ sets the upright value
        Raises/Assertions:
            if upright is not Boolean"""
        assert isinstance(upright, bool), "upright should be Boolean"
        self._upright = upright
        self.extractor.setBool('upright', upright)

    @property
    def extended(self):
        """returns the extended value"""
        return self._extended

    @extended.setter
    def extended(self, extended):
        """sets the extended value
        Raises/Assertions:
            if extended is not Boolean"""
        assert isinstance(extended, bool), "extended should be Boolean"
        self._extended = extended
        self.extractor.setBool('extended', extended)

    @property
    def hessian_thresh(self):
        """returns hessian_thresh value"""
        return self._hessian_thresh

    @hessian_thresh.setter
    def hessian_thresh(self, hessian_thresh):
        """sets the hessian_thresh value
        Raises/Assertions:
            if hessian_thresh is not float"""
        assert isinstance(
            hessian_thresh, float), "hessian_thresh should be Double"
        self._hessian_thresh = hessian_thresh
        if self.keypoint_detector.detect_keypoints:
            self.keypoint_detector.detector.setDouble(
                'hessianThreshold', hessian_thresh)

    def __getstate__(self):
        state = super(SurfExtractor, self).__getstate__()
        local_state = {
            'extended': self.extended,
            'upright': self.upright,
            'hessianThreshold': self.hessian_thresh,
            'keypoint_limit': self.keypoint_limit

        }
        state.update(local_state)
        return state

    def __setstate__(self, state):
        super(SurfExtractor, self).__setstate__(state)
        self.upright = state.get('upright', False)
        self.extended = state.get('extended', False)
        self.hessian_thresh = state.get('hessianThreshold', 100.0)
        self.keypoint_limit = state.get('keypoint_limit', None)

    def extract(self, list_of_imagefiles):
        """
        Returns: a list of tuples containing keypoints and descriptors for each
        image in the list.
        Each tuple consists of: (list_of_cv2.KeyPoints, array_with_descriptors)
        Each row in the array contains the descriptor of the corresponding KeyPoint
        """
        all_features = super(SurfExtractor, self).extract(list_of_imagefiles)
        if self.keypoint_limit:
            assert isinstance(
                self.keypoint_limit, int), "keypoint_limit should be Integer"
            for idx, feature in enumerate(all_features):
                (kpts, descriptr) = feature
                if len(kpts) > self.keypoint_limit:
                    kpts = kpts[0: self.keypoint_limit]
                    descriptr = descriptr[0: self.keypoint_limit]
                    all_features[idx] = (kpts, descriptr)
        return all_features


class LocalFeatureDetector(object):

    def __init__(self, feature_type, keyp_grid_params=None):
        """
        Initialize key point detector.
            If keyp_grid_params is empty, the keypoint detection algorithm is used.
            Otherwise, the params given are used to compute a fixed-grid of keypoint locations
        Args:
            feature_type: string (it has to be 'SURF' or 'BRISK')
            keyp_grid_params = [fix_im_width, fix_im_height, grid_interval]
                parameters for the grid of fixed keypoint locations
                if it's empty, locations and automatically detected by the algorithm
                e.g. keyp_grid_params = [320, 240, 18]
                Resize images to be (240 rows x 320 columns) and create a list of keypoint locations
                equally spaced in the image with an interval of 18 pixels.

        Returns:

        Raises/Assertions:
            Asserts if the feature type is not 'SURF' or 'BRISK'
            Asserts if the keyp_grid_params has a wrong number of elements
                (it should be empty or have 3 positive numbers)
        """
        assert feature_type in ['SURF', 'BRISK'], \
            "Feature type not supported"

        assert not keyp_grid_params \
            or (len(keyp_grid_params) == 3 and np.min(keyp_grid_params) > 0), \
            "Keypoint_grid_params list should be empty or contain 3 positive numbers"

        self.fixed_img_keypoints = None
        self.detector = None
        self.FIX_IM_WIDTH = None
        self.FIX_IM_HEIGHT = None
        self.grid_interval = None
        self.detect_keypoints = True

        if keyp_grid_params:
            self.FIX_IM_WIDTH = keyp_grid_params[0]
            self.FIX_IM_HEIGHT = keyp_grid_params[1]
            self.grid_interval = keyp_grid_params[2]

            self.fixed_img_keypoints = self.calculate_grid_of_keypoints()
            self.detect_keypoints = False

            logger.info('New Detector using fixed-location keypoints')
        else:
            self.detector = cv2.FeatureDetector_create(feature_type)
            logger.info('New Detector using automaticaly detected keypoints')

    def get_keypoints(self, img_gray):
        """
        Args:
            img_gray, 2-d array image previously loaded with cv2.imread (in gray scale)
            If the image passed is not like this, the detection returns None
        Returns:
            list of keypoints (cv2.KeyPoint) for the input image

        Raises/Assertions:
        """

        if self.detect_keypoints:
            return self.detector.detect(img_gray)
        else:
            return self.fixed_img_keypoints

    def calculate_grid_of_keypoints(
        self, h=None, w=None, intv=None, nLayer=1, size=25, angle=175
    ):
        """
        Helper function to compute a grid of fixed locations within an image.
        Returns the list of keypoints (cv2.Keypoint) defined for an image
        of the given dimensions (h x w).
        If no dimensions or interval intv are given, it uses self FIX_IM_HEIGHT, FIX_IM_WIDTH, grid_interval

        Args:
            h: image height
            w: image width
            intv: the distance/interval, in pixels, between keypoints
            nlayer: the number of layers inside a pyramid
            size: diameter of the meaningful keypoint neighborhood
            angle: computed orientation of the keypoint range 0-360 (-1 if not applicable)

        Returns: list of cv2.Keypoints

        Assertions/Exception:
        """
        keypoints = []
        response = 0

        if not h:
            h = self.FIX_IM_HEIGHT
        if not w:
            w = self.FIX_IM_WIDTH
        if not intv:
            intv = self.grid_interval

        for i in range(nLayer):
            currIntv = intv * (i + 1)
            currSize = size * (i + 1)

            for x in range(currSize, w - currSize, currIntv):
                for y in range(currSize, h - currSize, currIntv):
                    kp = cv2.KeyPoint(x, y, currSize, angle, response, 1, -1)
                    keypoints = keypoints + [kp]

        return keypoints


# BAG of WORDS
class BagOfWords(AbstractFeature):

    def __init__(self, feat_ext, vocabsize=None, vocab=None, projection=None):
        self.feat_ext = feat_ext
        assert hasattr(feat_ext, 'extract')
        self.projection = projection
        if self.projection:
            assert hasattr(projection, 'train')
            assert hasattr(projection, 'project')
        if vocab is not None:
            self.vocabsize = len(vocab)
        else:
            assert vocabsize, "Vocabsize required for training"
            self.vocabsize = vocabsize
        self.vocab = vocab

    def _learn_projection(self, all_feats):
        self.projection.train(all_feats)

    def _project(self, image_descs):
        projected_descs = []
        for i in image_descs:
            # TODO: Can project all together
            if i.descriptors is not None:
                proj_feats = self.projection.project(i.descriptors)
                desc = ImageDesc(i.cv_keypoints, proj_feats)
            else:
                desc = ImageDesc(i.cv_keypoints, None)
            projected_descs.append(desc)

        return projected_descs

    def _learn_vocab(self, all_feats):
        self._vocab_trainer = Vocabulary(self.vocabsize)
        self.vocab = self._vocab_trainer.train(all_feats)
        return self.vocab

    def _quantize(self, image_descs):
        codebooks = []
        for idesc in image_descs:
            if idesc.descriptors is not None:
                code, distortion = vq(idesc.descriptors, self.vocab)
            else:
                code = []
            codebooks.append(code)

        hists = self._build_histogram(image_descs, codebooks)
        return image_descs, hists

    def _build_histogram(self, image_descs, codebooks):
        hists = []
        for code in codebooks:
            hist = np.histogram(code, bins=range(0, len(self.vocab)+1))[0]
            s = hist.sum()
            hist_nmz = hist*(1.00/(s or 1))
            hists.append(hist_nmz)
        return np.asarray(hists)

    def _feature_extract(self, image_list):
        # extract feats for each image
        results = self.feat_ext.extract(image_list)
        image_descs = [ImageDesc(*r) for r in results]
        return image_descs

    def _merge_descs_to_feats(self, image_descs):
        return np.concatenate([i.descriptors for i in image_descs if i.descriptors is not None])

    def train(self, image_list):
        image_descs = self._feature_extract(image_list)
        # learn projection matrix
        if self.projection:
            all_feats = self._merge_descs_to_feats(image_descs)
            self._learn_projection(all_feats)
            image_descs = self._project(image_descs)
        # send projected feats for vocab learning
        if self.vocab is None:
            all_feats = self._merge_descs_to_feats(image_descs)
            self.vocab = self._learn_vocab(all_feats)
        # return quantized features for training images
        return self._quantize(image_descs)

    def extract(self, image_list):
        # feats : list of numpy arrays
        # assert self._is_trained
        image_descs = self._feature_extract(image_list)
        if self.projection:
            image_descs = self._project(image_descs)
        return self._quantize(image_descs)

    def save_feat_ext(self, file_path):
        with open(file_path, 'w') as f:
            pickle.dump(self.feat_ext, f)

    def save_to_dir(self, model_dir):
        assert self.vocab is not None, "vocab is not set"
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        class_dict = {}

        if self.projection:
            projection_file = self._proj_path(model_dir)
            self.projection.save_to_file(projection_file)
            class_dict['project'] = self.projection.__class__

        vocab_path = self._vocab_path(model_dir)
        np.save(vocab_path, self.vocab)

        feat_ext_file = self._feat_path(model_dir)
        self.feat_ext.save_to_file(feat_ext_file)
        class_dict['feat_ext'] = self.feat_ext.__class__

        # store classes for all instance objects
        # so that they can be loaded correctly in the future
        class_file = self._class_path(model_dir)
        with open(class_file, 'w') as f:
            pickle.dump(class_dict, f)

    @staticmethod
    def get_state(model_dir):
        cls = BagOfWords
        class_file = cls._class_path(model_dir)
        with open(class_file, 'r') as f:
            class_dict = pickle.load(f)

        feat_ext_file = cls._feat_path(model_dir)
        feat_ext = class_dict['feat_ext'].load_from_file(feat_ext_file)

        projection = None
        if 'project' in class_dict:
            proj_file = cls._proj_path(model_dir)
            projection = class_dict['project'].load_from_file(proj_file)

        vocab_file = cls._vocab_path(model_dir)
        vocab = np.load(vocab_file)
        return dict(vocab=vocab, vocabsize=len(vocab), feat_ext=feat_ext,
                    projection=projection)

    @classmethod
    def load_from_dir(cls, model_dir):
        state = cls.get_state(model_dir)
        obj = cls.__new__(cls)
        vars(obj).update(state)
        return obj

    @staticmethod
    def _class_path(model_dir):
        return os.path.join(model_dir, 'class_file')

    @staticmethod
    def _proj_path(model_dir):
        return os.path.join(model_dir, 'project')

    @staticmethod
    def _vocab_path(model_dir):
        return os.path.join(model_dir, 'vocab.npy')

    @staticmethod
    def _feat_path(model_dir):
        return os.path.join(model_dir, 'feat_ext')


class SpatialBagOfWords(BagOfWords):
    _attrs_to_save = ['n_levels', 'im_height', 'im_width']

    def __init__(self, n_levels, im_height, im_width, feat_ext, vocabsize=None,
                 vocab=None, projection=None):
        super(SpatialBagOfWords, self).__init__(feat_ext, vocabsize=vocabsize,
                                                vocab=vocab, projection=projection)
        self.n_levels = n_levels
        self.im_height = im_height
        self.im_width = im_width

    def _build_histogram(self, image_descs, codebooks):
        from affine.detection.vision.utils.scene_functions import compute_spatial_features
        if image_descs:
            # keypoints are the same for all images
            cv_kps = image_descs[0].cv_keypoints
            # keypoint.pt is (row, col) but we need (col, row)
            kps = np.asarray([cv_kp.pt[::-1] for cv_kp in cv_kps])
            return compute_spatial_features(np.asarray(codebooks), kps,
                                            len(self.vocab), self.n_levels,
                                            self.im_width, self.im_height)
        return []

    def save_to_dir(self, model_dir):
        super(SpatialBagOfWords, self).save_to_dir(model_dir)
        params = {attr: getattr(self, attr) for attr in self._attrs_to_save}
        with open(self._params_path(model_dir), 'w') as f:
            pickle.dump(params, f)

    @classmethod
    def get_state(cls, model_dir):
        state = super(SpatialBagOfWords, cls).get_state(model_dir)
        with open(SpatialBagOfWords._params_path(model_dir), 'r') as f:
            params = pickle.load(f)
        state.update(params)
        return state

    @staticmethod
    def _params_path(model_dir):
        return os.path.join(model_dir, 'params')


class Vocabulary(object):

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self._bin_dir = config.bin_dir()

    def _read_vocab(self, vocab_file):
        with open(vocab_file+'.desc', 'r') as f:
            num_floats = self.dim*self.vocab_size
            float_array = np.asarray(struct.unpack('%sf' % num_floats,
                                                   f.read()))
            centers = np.reshape(float_array, (self.vocab_size, self.dim))
        return centers

    def _write_xml(self, xml_file, cols, num_desc):
        with open(xml_file, 'w') as out:
            w = XMLGenerator(out=out, encoding='utf-8')
            w.startDocument()
            w.startElement("opencv_storage", {})
            w.startElement("cols", {})
            w.characters(str(cols))
            w.endElement("cols")
            w.startElement("nDescriptors", {})
            w.characters(str(num_desc))
            w.endElement("nDescriptors")
            w.endElement("opencv_storage")
            w.endDocument()

    def _write_to_file(self, feats):
        h, feat_file = mkstemp()
        os.close(h)
        feat_file += '.xml'
        desc_file = feat_file + '.desc'
        num_desc, cols = feats.shape
        self._write_xml(feat_file, cols, num_desc)
        feats.astype(np.float32).tofile(desc_file)
        return feat_file

    def train(self, feats):
        from affine.detection.vision.utils import bovw_functions
        h, vocab_file = mkstemp()
        os.close(h)
        vocab_file += '.xml'
        feat_file = self._write_to_file(feats)
        self.dim = feats.shape[1]
        bovw_functions.vocab_kms(self._bin_dir, feat_file,
                                 self.vocab_size, vocab_file)
        self.vocab = self._read_vocab(vocab_file)
        # FIXME: Delete the temp .xml, .desc files
        return self.vocab

    def save_to_file(self, file_path):
        assert self.vocab is not None, "Vocabulary is missing"
        with open(file_path, 'w') as f:
            np.save(f, self.vocab)

    @classmethod
    def load_from_file(cls, file_path):
        with open(file_path, 'r') as f:
            vocab = np.load(f)
        obj = cls(len(vocab))
        obj.vocab = vocab
        return obj


# BOUNDING BOXES
class SlidingWindowBoxExtractor(AbstractFeature):

    def __init__(self, scales=[1], patch_shapes=[(64, 64)],
                 step_size=0.5, center_area_offset=0, corner_area_sz=None, raise_on_size=True):
        """
        Class to extract sliding window patches.

        We can specify in which regions of the image (central, corners, both)
        we want to run the sliding window

        Args:
            scales: list of float,
                represents the scales factors we want to consider
            patch_shapes: list of int tuples,
                each tuple is a patch size (height, width) in pixels
            step_size: float,
                if >=1: fixed number of pixels in each step
                if <1: intersection allowed between consecutive windows
                       (ratio of bbox width and height)
            center_area_offset: int, size in pixels of margins to be ignored
                when running sliding window in the center of the image.
                If = 0, means we run sliding window in the whole image
                If None, means we do not consider a central region at all
            corner_area_sz: tuple, (height, width) in pixels of the corner regions where 
                to run sliding window.
                If None, we do not consider corner regions
            raise_on_size: if True, there would be an assert if size of center area or corner areas in the box extractor
                are greater than the actual image dimensions given

        """
        self.scales = scales
        self.patch_shapes = patch_shapes
        self.step_size = step_size
        self.center_area_offset = center_area_offset
        self.corner_area_sz = corner_area_sz
        self.raise_on_size = raise_on_size

    def _extract_bb_in_region(self, top_left_corner, bottom_right_corner):
        """
        Runs sliding window in the specified area defined by the coordinates

        Args:
            top_left_corner: (x,y) top left corner of region considered
            bottom_right_corner: (x,y) bottom right corner of region considered
        """
        patch_map = []
        patch_sizes = np.array(self.scales)
        step_x = self.step_size
        step_y = self.step_size

        all_shapes = []
        for s in patch_sizes:
            for height, width in self.patch_shapes:
                all_shapes += [(int(s * height), int(s * width))]
        all_shapes = set(all_shapes)

        x_ini, y_ini = top_left_corner
        x_end, y_end = bottom_right_corner
        for h, w in all_shapes:
            if self.step_size < 1:
                step_y = max(1, int(h * (1 - self.step_size)))
                step_x = max(1, int(w * (1 - self.step_size)))
            for y in range(y_ini, y_end - h + 1, step_y):
                for x in range(x_ini, x_end - w + 1, step_x):
                    patch_map.append([h, w, y, x])
        return patch_map

    def _get_corner_regions(self, im_w, im_h):
        """
        Return top left and bottom right coordinates of each of the corner
        patches in the given image size
        """
        corner_height = self.corner_area_sz[0]
        corner_width = self.corner_area_sz[1]
        tl_corner = [(0, 0), (corner_width, corner_height)]
        tr_corner = [(im_w - corner_width, 0),
                     (im_w, corner_height)]
        bl_corner = [(0, im_h - corner_height),
                     (corner_width, im_h)]
        br_corner = [(im_w - corner_width, im_h - corner_height),
                     (im_w, im_h)]

        return [tl_corner, tr_corner, bl_corner, br_corner]

    def _get_centered_region(self, im_w, im_h):
        """
        Return centered region coordinates given the input image dimensions
        """
        centered_region = \
            [(self.center_area_offset, self.center_area_offset),
             (im_w - self.center_area_offset, im_h - self.center_area_offset)]

        return centered_region

    def get_bboxes_in_image(self, im_w, im_h):
        """
        Return list of bounding box positions for an image of the given image dimensions

        Args:
            im_w: int, image width
            im_h: int, image hight
        Returns:
            list of of 4-int lists. Each of these 4 elements is of the form
                [height, width, top_left_y, top_left_x]
        Assertions:
            Asserts if raise_on_size is True and size of center area or corner areas in the box extractor
            are greater than the actual image dimensions given
        """
        corner_flag = True
        center_flag = True
        if self.center_area_offset and (self.center_area_offset > im_w or self.center_area_offset > im_h):
            if self.raise_on_size:
                raise AssertionError("offset to start placing boxes can't be greater than image size")
            else:
                center_flag = False
        if self.corner_area_sz and (self.corner_area_sz[1] >im_w or self.corner_area_sz[0] > im_h):
            if self.raise_on_size:
                raise AssertionError("corner area dimension can't be greater than image dimension")
            else:
                corner_flag = False

        all_bboxes = []
        regions_considered = []
        if self.corner_area_sz and corner_flag:
            regions_considered = self._get_corner_regions(im_w, im_h)
        if self.center_area_offset is not None and center_flag:
            regions_considered += [self._get_centered_region(im_w, im_h)]

        for region_params in regions_considered:
            top_left, bottom_right = region_params
            all_bboxes += self._extract_bb_in_region(top_left, bottom_right)

        return all_bboxes

    def extract(self, list_image_paths):
        """
        Given a list of images, obtain bboxes for all of them

        NOTE: this function assumes that the configuration of the extractor is
        feasible for the given images (i.e., the size of corner and center area
        is smaller or equal than the actual images passed)
        """
        all_bboxes = {}
        for indx, im_file in enumerate(list_image_paths):
            img = cv2.imread(im_file)
            im_h, im_w = img.shape[:2]
            current_image_bb = self.get_bboxes_in_image(im_w, im_h)
            all_bboxes[indx] = current_image_bb
        return all_bboxes


class GridBoxExtractor(SlidingWindowBoxExtractor):

    def __init__(self, grid_shape=[2,2]):
        """
        Class to extract grid patches.

        Args:
            grid_shape: tuple of height and width for the required grid
        """
        super(GridBoxExtractor, self).__init__(scales=[1],
                 step_size=0, patch_shapes=None,
                 center_area_offset=0, corner_area_sz=None,
                 raise_on_size=True)
        self.grid_shape = grid_shape

    def _extract_bb_in_region(self, top_left_corner, bottom_right_corner):
        """
        Returns a grid of patches for the specified ared

        Args:
            top_left_corner: (x,y) top left corner of region considered
            bottom_right_corner: (x,y) bottom right corner of region considered
        """
        patch_list = []

        x_ini, y_ini = top_left_corner
        x_end, y_end = bottom_right_corner
        step_y = (y_end - y_ini) / self.grid_shape[0]
        step_x = (x_end - x_ini) / self.grid_shape[1]
        h = step_y
        for i in range(self.grid_shape[0]):
            w = step_x
            if i == self.grid_shape[0] - 1:
                h += (y_end - y_ini) % self.grid_shape[0]
            for j in range(self.grid_shape[1]):
                if j == self.grid_shape[1] - 1:
                    w += (x_end - x_ini) % self.grid_shape[1]
                y = i * step_y
                x = j * step_x
                patch_list.append([h, w, y, x])
        return patch_list


class EdgeBoxExtractor(AbstractFeature):

    def __init__(self, patch_shapes=[(32, 100)], scales=[1],
                 alpha=0.65, min_group_size=8, score_thresh=5, mag_th=0.1):
        """
        Params:
            patch_shapes: dimensions of each bounding box patch. This should
            only include different aspect ratios, different scales will be
            handled by the scales parameter. (list of tuples)
            scales: scales of image to search (list of floats)
            alpha: param measuring how much overlap we require between
                the bounding box and ground truth. (float between 0 and 1)
            min_group_size: min number of pixels in an edge group (int)
            score_thresh: threshold for box scores (float)
            mag: matrix the same size as image of magnitudes of each pixel
            orient: matrix of orientations for each pixel
            LK: data structure for each row/col for fast lookup
                (tuple of dicts)
            group_num: number of edge groups found (int)
            group_dict: dict mapping each pixel to a group and each group
                to a list of pixels. (dict)
            affinities: map of affinity scores between edge groups (nparray)
            tmp_dir: tmp dir for resized images and edge output (string)
            height: height of each image (int)
            width: width of each image (int)
            mag_th: threshold for pixel magnitude
        """
        self.patch_shapes = patch_shapes
        self.scales = scales
        self.alpha = alpha
        self.min_group_size = min_group_size
        self.score_thresh = score_thresh
        self.mag = None
        self.orient = None
        self.LK = None
        self.group_num = 0
        self.group_dict = {}
        self.affinities = None
        self.tmp_dir = mkdtemp()
        self.height = None
        self.width = None
        self.mag_th = mag_th

    def group_edges(self):
        """
        Params:
            mag_th: threshold for pixel magnitude

        Returns: a dictionary of each pixel in the img pointing to the group
        num it belongs to, and group_num pointing to the list of pixels.
        """
        self.group_num = 0
        self.group_dict = {}

        for x in range(self.height):
            for y in range(self.width):
                if self.mag[x][y] <= self.mag_th:
                    self.mag[x][y] = 0

        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                if (i, j) not in self.group_dict and self.mag[i][j] > 0:
                    path = self._traverse(i, j, self.orient[i][j])
                    if path:
                        self.group_num += 1
                        self.group_dict[self.group_num] = path
                        self.group_dict.update(dict(
                            (p, self.group_num) for p in path))

    def _traverse(self, i, j, last_orient, orient_thresh=math.pi/2):
        """Greedy algorithm to check 8-connected neighbors and add to edge
        group.
        """
        x = [1, 0, -1, 1, -1, 1, 0, -1]
        y = [1, 1, 1, 0, 0, -1, -1, -1]
        orient_sum = 0
        found_neigh = True
        path = [(i, j)]
        while orient_sum < orient_thresh and found_neigh:
            found_neigh = False
            for k in range(8):
                i_neigh = i+x[k]
                j_neigh = j+y[k]
                if i < self.height-1 and j < self.width-1 and \
                    self.mag[i_neigh, j_neigh] > 0 and orient_sum + abs(
                        last_orient - self.orient[i_neigh, j_neigh]) < \
                        orient_thresh and (i_neigh, j_neigh) not in path:
                    i = i_neigh
                    j = j_neigh
                    found_neigh = True
                    orient_sum += abs(last_orient - self.orient[i, j])
                    last_orient = self.orient[i, j]
                    path.append((i, j))
                    break
        if len(path) > self.min_group_size:
            return path
        else:
            return None

    def compute_affinity(self):
        """Generate map of affinities between all edge groups. """
        self.affinities = np.zeros((self.group_num, self.group_num))
        group_theta = np.zeros(self.group_num)
        group_x = []
        for num in range(1, self.group_num+1):
            group_theta[num-1] = np.mean(
                [self.orient[i] for i in self.group_dict[num]])
            group_x.append(np.mean(self.group_dict[num], axis=0))
        for num in range(1, self.group_num+1):
            neighboring_groups = []
            points = self.group_dict[num]
            for p in points:
                x, y = p
                for i in range(x-2, x+3):
                    for j in range(y-2, y+3):
                        if (i, j) in self.group_dict and (i, j) != p:
                            neighboring_groups.append(self.group_dict[(i, j)])
            neighboring_groups = set(neighboring_groups)
            for g in neighboring_groups:
                theta_ij = scipy.arctan2(group_x[g-1][1]-group_x[num-1][1],
                                         group_x[g-1][0]-group_x[num-1][1])
                aff = abs(
                    math.cos(group_theta[g-1]-theta_ij) *
                    math.cos(group_theta[num-1]-theta_ij))**2
                if aff > 0.05:
                    self.affinities[num-1, g-1] = aff

    def compute_box_scores(self, patch_map):
        """Compute object proposal score for each patch in patch_map.  Depends
        on the number of edge groups intersecting the box.
        sb: set of edge groups that overlap box b's boundary.
        sb_in: set of edge groups that overlap box b_in's boundary.
        b_in: box with width and height w/2, h/2 centered in box b.
        w_b: array of continuous values [0, 1] that indicate whether an edge
            group is wholly contained in b.
        s_in, s: integral images to speed computation of all m_i of edge groups
        intersecting or contained in box b/b_in.
        Returns:
            list of box scores (float)
        """
        all_Lr, all_Kr, all_Lc, all_Kc = self.LK
        box_scores = []
        for p in patch_map:
            h, w, x, y = p
            sb = np.unique(np.concatenate(
                (all_Lr[x][all_Kr[x][y:y+w]],
                 all_Lr[x+h-1][all_Kr[x+h-1][y:y+w]],
                 all_Lc[y][all_Kc[y][x:x+h]],
                 all_Lc[y+w-1][all_Kc[y+w-1][x:x+h]])))
            sb = sb[np.nonzero(sb)]
            s = np.array(
                [idx for idx in range(1, self.group_num+1) if
                 self.x_i[idx][0] > x and self.x_i[idx][0] < x+h and
                 self.x_i[idx][1] > y and self.x_i[idx][1] < y+w and
                 idx not in sb], dtype=int)
            x_in = x + h/4
            y_in = y + w/4
            w_in = w / 2
            h_in = h / 2
            sb_in = np.unique(np.concatenate(
                (all_Lr[x_in][all_Kr[x_in][y_in:y_in+w_in]],
                 all_Lr[x_in+h_in-1][all_Kr[x_in+h_in-1][y_in:y_in+w_in]],
                 all_Lc[y_in][all_Kc[y_in][x_in:x_in+h_in]],
                 all_Lc[y_in+w_in-1][all_Kc[y_in+w_in-1][x_in:x_in+h_in]])))
            s_in = np.array(
                [idx for idx in s if
                 self.x_i[idx][0] > x_in and self.x_i[idx][0] < x_in+h_in and
                 self.x_i[idx][1] > y_in and self.x_i[idx][1] < y_in+w_in and
                 idx not in sb_in], dtype=int)
            w_b = np.zeros(len(s))
            for i, s_i in enumerate(s):
                scores = self.affinity_paths[s_i-1, sb-1]
                max_score = 0
                if len(scores) > 0:
                    max_score = np.max(scores)
                w_b[i] = 1 - max_score
            h_b = np.sum(w_b*self.m_i[s-1]) / (2*(h+w)**1.5)
            h_bin = h_b - np.sum(self.m_i[s_in-1]) / (2*(h+w)**1.5)
            box_scores.append(h_bin)
        return box_scores

    def get_affinity_map(self):
        """Compute affinity paths using Dijkstra's alg. and x_i and m_i
        matrices.

        x_i: a list of one pixel/edge group to check if the edge group is
            inside the bounding box or not.
        m_i: list of sums of magnitudes for each edge group.
        """

        from scipy.sparse.csgraph import dijkstra
        affinity_graph = self.affinities.copy()
        nonzero = np.nonzero(affinity_graph)
        affinity_graph[nonzero] = - np.log(affinity_graph[nonzero])
        self.affinity_paths = - dijkstra(affinity_graph, directed=False)
        nonzero = np.nonzero(self.affinity_paths)
        self.affinity_paths[nonzero] = np.exp(self.affinity_paths[nonzero])
        self.x_i = dict((num, self.group_dict[num][0]) for num in range(
            1, self.group_num + 1))
        self.m_i = np.array([np.sum([self.mag[i] for i in self.group_dict[s]])
                             for s in range(1, self.group_num+1)])

    def get_sliding_window_patches(self):
        """Patch_map is a list of patches, with each patch represented as a
        tuple with the image number, (height, width), x, y (upper left)
        """
        bb_extractor = SlidingWindowBoxExtractor(
            scales=self.scales, patch_shapes=self.patch_shapes,
            step_size=(math.sqrt(self.alpha)))
        patch_map = bb_extractor.get_bboxes_in_image(im_w=self.width,
                                                     im_h=self.height)
        return patch_map

    def get_LK(self):
        """Build and return two arrays for each row/col of an image.
        Builds:
            all_Lc, all_Lr: Ordered list of edge group indices
            all_Kc, all_Kr: List of indices into L for each pixel in the row.
        """
        all_Lr = []
        all_Kr = []
        all_Lc = []
        all_Kc = []
        for i, row in enumerate(self.mag):
            Lr = []
            Kr = np.zeros(self.width, dtype=int)
            for r in range(self.width):
                if (i, r) in self.group_dict:
                    g = self.group_dict[(i, r)]
                else:
                    g = 0
                if Lr == [] or Lr[-1] != g:
                    Lr.append(g)
                Kr[r] = len(Lr) - 1
            all_Lr.append(np.array(Lr, dtype=int))
            all_Kr.append(Kr)
        for j, col in enumerate(self.mag.T):
            Lc = []
            Kc = np.zeros(self.height, dtype=int)
            for c in range(self.height):
                if (c, j) in self.group_dict:
                    g = self.group_dict[(c, j)]
                else:
                    g = 0
                if Lc == [] or Lc[-1] != g:
                    Lc.append(g)
                Kc[c] = len(Lc) - 1
            all_Lc.append(np.array(Lc, dtype=int))
            all_Kc.append(Kc)
        self.LK = (all_Lr, all_Kr, all_Lc, all_Kc)

    def refine_boxes(self, boxes, scores):
        """Greedy iterative search to maximize box score.

        Params: list of boxes and their box scores
        Returns: new list of refined boxes and their box scores
        """
        step_size_ratio = (1-self.alpha) / (1+self.alpha)
        for i, b in enumerate(boxes):
            h, w, x, y = b
            r_step = int(h*step_size_ratio)/2
            c_step = int(w*step_size_ratio)/2
            idx = None
            while r_step > 2 or c_step > 2:
                r_step = max(1, r_step)
                c_step = max(1, c_step)
                temp_boxes = [[h, w+c_step, x, y],
                              [h, w, x, y+c_step],
                              [h+r_step, w, x, y],
                              [h, w, x+r_step, y]]
                new_boxes = []
                for b in temp_boxes:
                    h1, w1, x1, y1 = b
                    if x1 + h1 < self.height and y1 + w1 < self.width:
                        new_boxes.append(b)
                box_scores = self.compute_box_scores(new_boxes)
                new = len(new_boxes)
                box_scores = np.array(box_scores + [scores[i]])
                max_score = box_scores.max()
                idx = box_scores.argmax()
                r_step /= 2
                c_step /= 2
            if idx and idx != new:
                boxes[i] = new_boxes[idx]
                scores[i] = max_score
        return boxes, scores

    def resize_image(self, image, length=400):
        """Resize image so the longer side is of size length or less."""
        m, n = image.shape[:2]
        longer = max(m, n)
        scale = 1
        if longer > length:
            scale = length / float(longer)
            new_m = int(m*scale)
            new_n = int(n*scale)

            return cv2.resize(image, (new_n, new_m)), scale
        return image, scale

    def draw_boxes(self, img_files, all_boxes, out_dir):
        """Draws boxes in images and saves them to out_dir.

        Parameters:
            img_files: list of file paths
            all_boxes: output from EdgeBoxExtractor.extract()
            out_dir: string directory
        """
        for idx, im_file in enumerate(img_files):
            img = cv2.imread(im_file)
            boxes = all_boxes[idx]
            for b in boxes:
                h, w, y, x = b
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(out_dir, os.path.split(im_file)[1]), img)

    def extract(self, img_files, return_all_boxes=False):
        """Returns edge boxes for images.  If boxes are too large, resize them,
        calculate boxes, then resize boxes back to original img size.

        Params: list of image file paths (string)
                return_all_boxes: boolean flag to return all boxes with scores
                instead of thresholding, refining, and using NMS.

        Assertions: Checks files in img_files exist

        Returns:
            all_boxes: dict with image filename position in the input list as key
                and list of boxes as [h, w, x, y] as value.
            If return_all_boxes is True, all_boxes values are a tuple of list
            of boxes and box scores.
        """
        all_boxes = {}
        for idx, im_file in enumerate(img_files):
            assert os.path.isfile(im_file), "File %s does not exist." % im_file
            image = cv2.imread(im_file)
            if image is not None:
                image, scale = self.resize_image(image)
                self.height, self.width = image.shape[:2]
                im_path = os.path.join(self.tmp_dir, 'img.jpg')
                cv2.imwrite(im_path, image)
                edges = structured_edge_detection(im_path)
                if np.sum(edges) > 0:
                    self.mag, self.orient = edge_nms(edges)
                    self.group_edges()
                    self.compute_affinity()
                    self.get_LK()
                    self.get_affinity_map()
                    patch_map = self.get_sliding_window_patches()
                    scores = self.compute_box_scores(patch_map)
                    if return_all_boxes:
                        all_boxes[idx] = (np.array(
                            patch_map) / scale).astype(int), scores
                        continue
                    patch_map = [pm for i, pm in enumerate(patch_map) if
                                 scores[i] > self.score_thresh]
                    scores = [s for s in scores if s > self.score_thresh]

                    refined_boxes, refined_scores = self.refine_boxes(
                        patch_map, scores)

                    # Use NMS to eliminate boxes
                    sorted_boxes = [x for y, x in sorted(zip(
                        refined_scores, refined_boxes), reverse=True)]
                    all_boxes[idx] = (np.array(box_nms(sorted_boxes, (
                        self.height, self.width))) / scale).astype(int)
        return all_boxes


class BoundingBoxRegressor(AbstractFeature):

    def __init__(self, model_name):
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        """Load r-CNN Model"""
        from affine.detection.model_worker.tasks.data_processor_task import DataProcessorClient
        self.r_cnn = DataProcessorClient(self.model_name)

    def train_model(self, input_images, p_boxes, g_boxes, folds=5):
        """Learn parameters for bounding box regression.

        Args:
            input_images: list of cropped image input files to r-cnn. The input
            images should be centered and 2*h by 2*w of the orig bounding box.
            p_boxes: numpy array of h, w, midx, midy
            g_boxes: numpy array of h, w, midx, midy of ground truth boxes
            folds: number of cross_val folds

        blobs: features from pool5 layer of r-cnn

        Returns:
            score: cross validation score
            list of model coefficients (w_h, w_w, w_x, w_y) for
        regression (in opencv params) -> (h, w, y, x) in numpy
        """
        blobs = self.get_features(input_images)
        clf = Ridge(alpha=1000)

        total_x = [blobs] * 4
        total_y = [np.log(g_boxes[:, 0] / p_boxes[:, 0]),
                   np.log(g_boxes[:, 1] / p_boxes[:, 1]),
                   (g_boxes[:, 2] - p_boxes[:, 2]) / p_boxes[:, 0],
                   (g_boxes[:, 3] - p_boxes[:, 3]) / p_boxes[:, 1]]
        coefficients = []
        scores = []
        kf = KFold(len(p_boxes), n_folds=folds)
        for i in range(4):
            x = total_x[i]
            y = total_y[i]
            coefficients.append(clf.fit(x, y).coef_)
            cv_scores = []
            for train, test in kf:
                train_x = x[train]
                train_y = y[train]
                test_x = x[test]
                test_y = y[test]
                clf.fit(train_x, train_y)
                cv_scores.append(clf.score(test_x, test_y))
            scores.append(np.mean(cv_scores))
        return coefficients, scores

    def get_features(self, image_list):
        """Extract pool5 layers from R-CNN.

        Params: image_list: list of strings of file paths
        """
        from affine.detection.model_worker.tasks.data_processor_task import convert_files_to_bin
        image_list = batch_list(image_list, batch_size=1)
        output = []
        for batch in image_list:
            output.append(self.r_cnn.predict(convert_files_to_bin(batch)))
        merged_result = merge_list(output)
        pool5 = np.array(merged_result)
        return pool5

    def extract(self, coefficients, bounding_boxes, img_files):
        """Return regression output for bounding boxes.  Coeffs are ordered by
        img coord (h, w, x, y) -> numpy coord (h, w, y, x).  The R_CNN expects
        box input in (ymin, xmin, ymax, xmax) format of a window with 2*w and
        w*h centered at the original box. The regression model outputs
        in (h, w, midy, midx) format.

        Params:
            bounding_boxes: list of img index and box size: (i, h, w, x, y)
            coefficients: numpy array of coefficients for h, w, x, y linear
            models
            img_files: list of strings

        Returns: new_bounding_boxes: list of img index and box size
        """
        total_boxes = np.zeros((len(bounding_boxes), 4), dtype=int)
        box_ind_dict = defaultdict(list)
        new_box_dict = defaultdict(list)
        for i, pm in enumerate(bounding_boxes):
            idx, h, w, x, y = pm
            box = [h, w, x, y]
            new_box = [max(0, x-h/2), max(0, y-h/2), x+3*h/2, y+3*w/2]
            total_boxes[i, :] = box
            box_ind_dict[idx].append(i)
            new_box_dict[idx].append(new_box)

        images_windows = [
            (img_files[im], new_box_dict[im]) for im in sorted(
                new_box_dict.keys())]
        window_inputs = bbu.crop_windows(images_windows)
        img_files = bbu.write_to_dir(window_inputs)
        pool5 = self.get_features(img_files).transpose()
        predicted_edge_boxes = np.zeros(total_boxes.shape, dtype=int)
        predicted_edge_boxes[:, 0] = total_boxes[:, 0] * np.exp(
            coefficients[0, :].dot(pool5))
        predicted_edge_boxes[:, 1] = total_boxes[:, 1] * np.exp(
            coefficients[1, :].dot(pool5))
        predicted_edge_boxes[:, 2] = total_boxes[:, 2] * \
            coefficients[3, :].dot(pool5) + total_boxes[:, 2]
        predicted_edge_boxes[:, 3] = total_boxes[:, 3] * \
            coefficients[2, :].dot(pool5) + total_boxes[:, 3]

        new_bounding_boxes = []
        for i in range(len(img_files)):
            inds = box_ind_dict[i]
            if inds:
                new_bounding_boxes += list(np.concatenate(
                    [np.ones((len(inds), 1), dtype=int)*i,
                     predicted_edge_boxes[inds, :]], axis=1))
        return new_bounding_boxes
