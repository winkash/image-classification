import os
import cv2
import cv2.cv as cv
import re
import numpy as np
import xml.etree.ElementTree as ET

from configobj import ConfigObj
from validate import Validator
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances

from affine.model import session, TrainingImage, Video

__all__ = ['sample_frames', 'get_params_from_config',
           'check_infofile', 'flip_image', 'download_images', 'download_images_to_dir',
           'make_infofile', 'get_config', 'read_image_file', 'save_training_images',
           'binary_to_ndarray', 'compute_spatial_features', 'compute_vocab_features',
           'extract_from_xml', 'average_scales', 'prefilt', 'pad_image_symmetric',
           'scale_filter', 'SCENE_CFG_SPEC', 'POS_LABEL', 'NEG_LABEL',
           'compute_spatial_features_from_xml']

POS_LABEL = 1
NEG_LABEL = -1

# Resized image sizes used in dense_feature_extract binary
IM_WIDTH = 800
IM_HEIGHT = 400

SCENE_CFG_SPEC = """
    [url_injection]
        query_string = string(min=3, max=400)
        max_vids = integer(min=1, max=5000, default=500)
        min_num_vids_downloaded = integer(min=100, max=5000, default=300)
        url_injection_priority = integer(default=30)
        time_out = integer(min=12, max=24, default=24)
    [mturk_submission_params]
        mturk_question = string(max=2000)
    [train_detector_params]
        pos_min_num_frames = integer(min=1, max=5000, default=1000)
        pos_optimal_num_frames = integer(min=1, max=5000, default=1000)
        neg_train_min_num_frames = integer(min=1, max=8000, default=5000)
        neg_test_min_num_frames = integer(min=0, max=1000, default=100)
        split_ratio = float(min=0.0, max=1.0, default = 0.8)
        detector_type = string(default='Scene')
        feature_type = string(default='SURFEX')
        pca_dimensions = integer(default=60)
        vocab_size = integer(min=500, default=500)
        keypoint_intv = integer(min=1, default=15)
        keypoint_nLayer = integer(min=1, default=1)
        keypoint_size = integer(min=1, default=25)
        keypoint_angle = integer(min=0, default=175)
        video_threshold = integer(min=1, default=2)
        image_threshold = float(default=0.2)
        svm_type = integer(default=3)
        svm_kernel = integer(default=5)
        smv_nu = float(default=0.2)
        svm_gamma = float(default=0.05)
        blocks = integer(default=4)
        orientations = int_list(default=list(8, 8, 8, 8))
        num_levels = integer(default=3)
    [model_to_db_injection]
        detector_name = string(min=3, max=50)
        target_label = string(min=3, max=50)
    """


def sample_frames(videos,  n_images, secs):
    """ Frame sampling from a video
        Args:
            videos: a list of video ids
            n_images: number of images (frames) to sample per video. If the number is bigger than the existing frames, we return all frames
            secs: sampling rate (int)
        Returns:
            set_videos: a list of tuples (video_id, timestamp)
        Assertions:
            AssertionError when n_images and secs <= 0, and when secs is not an integer
    """
    assert n_images > 0, "Number of images to sample should be > 0"
    assert secs > 0, "Sampling rate should be an integer > 0"
    assert isinstance(secs, int), "Sampling rate should be an integer"
    set_videos = []
    for video_id in videos:
        v = Video.get(video_id)
        timestamps = v.s3_timestamps()
        if not timestamps:
            continue
        slice_factor = len(timestamps)/ ((v.length/secs) or 1) or 1
        frames = timestamps[::slice_factor]
        if len(frames) > n_images:
            np.random.shuffle(frames)
            set_videos += zip([video_id]*n_images, frames[0:n_images])
        else:
            set_videos += zip([video_id]*len(frames), frames)
    return set_videos

def check_infofile(info):
    """ Checks that infofile can be read and it has the right format
        Args:
            info: path to infofile
        Assertions:
            IOError if file cannot be read
            AssertionError: if bad formatting
    """
    if not (os.path.isfile(info) and os.access(info, os.R_OK)):
            raise IOError('Failed trying to read file %s' % info)
    with open(info, 'r') as fo:
        lines = fo.read().splitlines()

    for l in lines:
        p, la, v, t = l.split(' ')
        assert isinstance(p, str), "Path is not a string"
        assert os.path.exists(p), "Path doesn't exist"
        assert int(la) == 1 or int(la) == 0 or int(la) == -1, "Wrong label format"
        assert re.search(r'^-?\d+$', v), "Video is not a number"
        assert t.isdigit(), "Timestamp is not a number"
    return True

def get_params_from_config(scene_params):
    """ Gets feature parameters from the configobj created from the config file
        Args:
            scene_params: configobj
        Returns:
            params: a dictionary with feature information for feature extraction
    """
    feature_type = scene_params['train_detector_params']['feature_type']
    if  feature_type == 'SURF' or feature_type == 'SURFEX':
        params = dict(
        # dense feature extraction params:
        in_intv = int(scene_params["train_detector_params"]["keypoint_intv"]),
        in_nLayer = int(scene_params["train_detector_params"]["keypoint_nLayer"]),
        in_size = int(scene_params["train_detector_params"]["keypoint_size"]),
        in_angle = int(scene_params["train_detector_params"]["keypoint_angle"]),
        )
    elif feature_type == 'HOG':
        params = dict(
        # HOG feature extraction params:
        hog_variant = scene_params["train_detector_params"]["hog_variant"],
        cell_size = scene_params["train_detector_params"]["cell_size"],
        num_orientations = scene_params["train_detector_params"]["num_orientations"],
        )
    elif feature_type == 'ColorSURFEX':
        params = dict(
        # color dense feature extraction params:
        in_color_type = scene_params["train_detector_params"]["color_type"],
        in_intv = int(scene_params["train_detector_params"]["keypoint_intv"]),
        in_nLayer = int(scene_params["train_detector_params"]["keypoint_nLayer"]),
        in_size = int(scene_params["train_detector_params"]["keypoint_size"]),
        in_angle = int(scene_params["train_detector_params"]["keypoint_angle"]),
        )
    elif feature_type == 'GIST':
        ori = list(scene_params["train_detector_params"]["orientations"])
        ori = map(int, ori)
        params = dict(
        # gist feature  params:
        orientations = ori,
        nblocks = int(scene_params["train_detector_params"]["blocks"]),
        )
    else:
        raise IOError("Unknown feature")

    params['in_feature_type'] = scene_params["train_detector_params"]["feature_type"]
    # PCA params:
    params['pca_dimensions'] = int(scene_params["train_detector_params"]["pca_dimensions"])
    # vocabulary params:
    params['vocab_size'] = int(scene_params["train_detector_params"]["vocab_size"])
    # SVM params:
    params['svm_type'] = int(scene_params["train_detector_params"]["svm_type"])
    params['svm_kernel'] = int(scene_params["train_detector_params"]["svm_kernel"])
    params['svm_nu'] = float(scene_params["train_detector_params"]["svm_nu"])
    params['svm_gamma'] = float(scene_params["train_detector_params"]["svm_gamma"])
    return params

def flip_image(src, dst):
    """ Reflects an image across y-axis
        Args:
            src: string, source path
            dst: string, destination path
    """
    im = cv2.imread(src)
    im_flipped = cv2.flip(im, 1)
    cv2.imwrite(dst, im_flipped)

def download_images(images, paths):
    """Download a collection of images.

    Args:
        images: List of (video id, timestamp)s.
        paths: Corresponding list of paths where the images will be downloaded.
    """
    for image, path in zip(images, paths):
        vid, ts = image
        Video.get(vid).download_image(ts, path)

def download_images_to_dir(images, image_dir):
    """Download a collection of images to a directory.
    
    Args:
        images: List of (video id, timestamp)s.
        image_dir: Download directory.
    
    Returns:
        List of paths to the downloaded images, in the same order as IMAGES.
    """
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    paths = [os.path.join(image_dir, '%s_%s.jpg' % image) for image in images]
    download_images(images, paths)
    return paths

def make_infofile(info_path, image_paths):
    """Make an infofile.

    Args:
        info_path: Path of infofile.
        image_paths: List of (video id, timestamp)s.
    """
    with open(info_path, 'w') as fo:
        for path in image_paths:
            line = '%s 0 0 0\n' % path
            fo.write(line)

def get_config(path, spec):
    """Get config.

    Args:
        path: Config file path.
        spec: configspec.

    Returns:
        A ConfigObj.

    Raises:
        AssertionError: Config does not meet specs.
    """
    config = ConfigObj(path, configspec=spec)
    result = config.validate(Validator(), copy=True, preserve_errors=True)
    assert result is True, 'Config file validation failed: %s' % result
    return config

def read_image_file(path):
    """Read image file.

    Each line in the image file must be of the form: video id<TAB>timestamp.

    Args:
        path: Path to image file.

    Returns:
        List of (video id, timestamp)s.
    """
    with open(path, 'r') as fo:
        lines = fo.read().splitlines()
    f = lambda line: tuple(int(x) for x in line.split('\t'))
    return map(f, lines)

def save_training_images(detector_id, images, labels):
    """Save collection of images used for training.

    Args:
        detector_id: Id of detector.
        images: List of (video id, timestamp)s.
        labels: List of labels.

    Raises:
        AssertionError
    """
    assert len(images) == len(labels)
    assert all([label in [POS_LABEL, NEG_LABEL] for label in labels])
    for (video_id, timestamp), label in zip(images, labels):
        TrainingImage(detector_id=detector_id, video_id=video_id,
                      timestamp=timestamp, label=label)
    session.flush()

def binary_to_ndarray(filename, dims, dtype):
    x = np.fromfile(filename, dtype=dtype)
    return x.reshape(dims)

def _get_spatial_histogram(vocab_feat, keypoints, vocab_size, num_levels,
                                 im_width, im_height, num_cells):
    num_images = vocab_feat.shape[0]
    # 3D mat where each layer represents the 2D feature mat for an image
    counts_mat = np.zeros((num_images, num_cells, vocab_size))
    image_range = np.arange(num_images)
    for kp_id, kp in enumerate(keypoints):
        # the vocab for this keypoint across all images
        vocab_for_kp = vocab_feat[:, kp_id]
        # a level consists of multiple cells
        # level 0 : 1 cell, level 1: 4 cells and so on
        # for eg : if num_levels = 3, total_cells = 21
        # cells start from level 0 and follow column-major order
        # each row represents the histogram for that cell
        col, row = kp[0], kp[1]
        # for every level add to appropriate cell/bucket
        # for level 0 special case
        counts_mat[image_range, 0, vocab_for_kp] += 1
        # from 1 level onwards,
        base_cell_count = 0
        for level in range(1, num_levels):
            # Figure out the cell id
            base_cell_count += 4**(level-1)
            row_div = im_height/(2.**level)
            col_div = im_width/(2.**level)
            grid_pos = (int(row/row_div), int(col/col_div))
            cell_id = base_cell_count + (2**level)*grid_pos[1] + grid_pos[0]
            counts_mat[image_range, cell_id, vocab_for_kp] += 1
    return counts_mat

def compute_spatial_features_from_xml(extract_path, projected_path, vocab_path,
                                      num_levels):
    """ A convenience function that computes spatial pyramid features directly 
    from the XML outputs of vision binaries.

    ${EXTRACT_PATH}.desc and ${PROJECTED_PATH}.desc must exist.

    Args:
        extract_path: Path to output of `dense_feature_extract`.
        projected_path: Path to output of `pca_projection`.
        vocab_path: Path to output of `vocab_kms`.
        num_levels: Number of levels.

    Returns:
        Ndarray with shape [NUM_IMAGES, NUM_FEATURES].
    """
    dims = extract_from_xml(vocab_path, ['numCenters', 'dimension'])
    vocab = binary_to_ndarray('%s.desc' % vocab_path, dims, np.float32)
    dims = extract_from_xml(projected_path, ['nDescriptors', 'cols'])
    proj = binary_to_ndarray('%s.desc' % projected_path, dims, np.float32)
    keypoints = np.asarray(cv.Load(extract_path, name='keypoints')).squeeze()
    im_height, im_width, num_images = extract_from_xml(extract_path,
                                                       ['image_height',
                                                       'image_width', 'nImg'])
    vocab_feat = compute_vocab_features(vocab, proj).reshape(num_images,
                                                             keypoints.shape[0])
    return compute_spatial_features(vocab_feat, keypoints, vocab.shape[0],
                                    num_levels, im_width, im_height)

def compute_spatial_features(vocab_feat, keypoints, vocab_size, num_levels,
                             im_width, im_height):
    """Compute spatial pyramid features.

    Args:
        vocab_feat: Ndarray with shape [NUM_IMAGES, NUM_KEYPOINTS].
        keypoints: Ndarray with shape [NUM_KEYPOINTS, 2].
        vocab_size: Size of vocabulary.
        num_levels: Number of levels.
        im_width: Image width.
        im_height: Image height.

    Returns:
        Ndarray with shape [NUM_IMAGES, NUM_FEATURES].
    
    Note:
        keypoints are (col, row) NOT (row, col).
    """
    # computing weights for each cell
    cell_weights = np.asarray([pow(2, 1-num_levels)] + sum([[pow(2, -i)]*pow(4, num_levels-i)
                                                 for i in xrange(num_levels-1, 0, -1)], []))
    num_cells = len(cell_weights)
    feature_length = num_cells*vocab_size
    num_images = vocab_feat.shape[0]
    # initializing the feature matrix of the right size,
    features = np.zeros((num_images, feature_length))
    # get a 3D mat where each layer represents the 2D feature mat for an image
    # the 2D mat is of the shape num_cells X vocab_size
    # where each row represents the histogram for the corresponding cell
    feature_mat = _get_spatial_histogram(vocab_feat, keypoints, vocab_size,
                            num_levels, im_width, im_height, num_cells)
    for image_id, mat in enumerate(feature_mat):
        weighted_mat = mat * cell_weights[:, np.newaxis]
        features[image_id, :] = np.hstack(weighted_mat)

    return features

def compute_vocab_features(vocab, feat):
    """Compute vocabulary features.

    Args:
        vocab: Ndarray with shape [NUM_WORDS, NUM_FEATURES]
        feat: Ndarray with shape [K, NUM_FEATURES]

    Returns:
        Ndarray with shape (K,)
    """
    # batch computation to avoid out of memory error on large datasets
    batch_size = vocab.shape[0]
    k = feat.shape[0]
    codes = np.zeros(k, dtype=np.int)
    for i in xrange(0, k, batch_size):
        sl = slice(i, i+batch_size)
        codes[sl] = np.argmin(euclidean_distances(feat[sl,:], vocab, squared=True), axis=1)
    return codes

def extract_from_xml(path, tags):
    """Extract integer content from xml.

    Elements must be children of the xml root.

    Args:
        path: path to xml file.
        tags: list of tags.

    Returns:
        List of the tags' integer content.

    Raises:
        AssertionError: Each tag must occur exactly once.
    """
    root = ET.parse(path).getroot()
    ret = []
    for tag in tags:
        res = root.findall(tag)
        assert len(res) >= 1, 'Not found: %s' % tag
        assert len(res) <= 1, 'Multiple tags: %s' % tag
        ret.append(int(res[0].text))
    return ret

def average_scales(img, blocks):
    """ compute averages over an grid of blocks x blocks
        Args:
            img: a numpay array image (grayscale)
            blocks: number of blocks
        Returns:
            y: numpy array image
        Assertions:
            AssertionError if the number of blocks <= 0
    """
    assert blocks > 0 , "number of blocks should be > 0 "
    if len(img.shape) == 2:
        img = np.tile(img, [1, 1, 1])
    rows = img.shape[1]
    cols = img.shape[2]
    ny = np.fix(np.linspace(0, rows, blocks + 1))
    nx = np.fix(np.linspace(0, cols, blocks + 1))
    y = np.zeros([1, blocks, blocks])
    for r in xrange(len(ny) - 1):
        for c in xrange(len(nx) - 1):
            bl = img[:, ny[r]:ny[r + 1], nx[c]:nx[c + 1]]
            bl = bl.reshape((1, -1)).mean(axis=1)
            y[:,r,c] = bl
    return y

def scale_filter(image_path, img_size):
    """ scales intensities and applies local contrast scaling over an image. It resizes the image to img_size x image_size
        Args:
            image_path: path to image
            img_size: size of the final image. The resulting image will be squared and grayscale
        Returns:
            pix: numpy array of the image
        Assertions:
            AssertionError if the path to the image is incorrect or the image doesn't exist
    """
    assert os.path.exists(image_path), "image doesn't exist"
    img = Image.open(image_path)
    img = img.convert('L')
    immax = max(img.size)
    immin = min(img.size)
    img = img.crop([int((immax/2.0) - (immin/2.0)), 0, int(immax/2.0 + immin/2.0), immin])
    img = img.resize([img_size, img_size], Image.ANTIALIAS)
    #scale intensities
    pix = np.array(img)
    pix = pix - np.amin(pix)
    maxp = np.amax(pix)
    if maxp:
        pix = 255.0*pix/maxp
    else:
        pix = 255.0*pix
    #local contrast scaling
    pix = prefilt(pix)
    return pix

def prefilt(pix):
    """ perform local contrast scaling on an image
        Args:
            pix: 2D numpay array
        Returns:
            final: 2D numpy array
        Assertions:
            AssertionError if the image/array is multidimensional
    """
    assert len(pix.shape) == 2, "prefilt only works for grayscale images"
    w = 5
    s1 = 4/np.sqrt(np.log(2))
    pix = np.log(pix + 1)
    bg = pad_image_symmetric(w, pix)
    y = np.linspace(-bg.shape[0]/2, (bg.shape[0]/2) - 1, bg.shape[0])
    x = np.linspace(-bg.shape[1]/2, (bg.shape[1]/2) - 1, bg.shape[1])
    xx, yy = np.meshgrid(x, y)
    gf = np.fft.fftshift(np.exp(-(pow(xx, 2) + pow(yy, 2))/pow(s1, 2)))
    #whitening (enhanced edges)
    gf = np.fft.ifft2(np.fft.fft2(bg)*gf)
    output = bg - np.real(gf)
    final = output[w:pix.shape[0] + w, w:pix.shape[1] + w]
    return final

def pad_image_symmetric(w, pix):
    """ pad an image symetrically
        Args:
            w: number of pixels to pad the image on each side
            pix: n x m numpy matrix
        Returns:
            bg: (n+w) x (m+w) numpy array
        Assertions:
            AssertionError: if image is multidemensional and if w <= 0
    """
    assert len(pix.shape) == 2, "pad_image_symmetric only supports grayscale images"
    assert w, "padding needs to be bigger than zero pixels"
    bg = np.zeros([pix.shape[0] + 2*w, pix.shape[1] + 2*w], dtype=pix.dtype)
    delta = (bg.shape[0] - pix.shape[0])/2
    bg[delta:pix.shape[0] + delta, delta:pix.shape[1] + delta] = pix
    flip = np.flipud(pix)
    bg[0:w, w:pix.shape[1] + w] = flip[pix.shape[0] - w:pix.shape[1], :]
    bg[pix.shape[0] + w:bg.shape[0], w:bg.shape[1] - w] = flip[0:w, :]
    flip = np.fliplr(pix)
    bg[w:pix.shape[0] + w, 0:w] = flip[:, pix.shape[1] - w:pix.shape[1]]
    bg[w:bg.shape[0] - w, pix.shape[1] + w:bg.shape[1]] = flip[:, 0:w]
    #fill small corners. This can be replaced when we upgrade numpy
    bg[0:w, 0:w] = bg[w:2*w, 0:w]
    bg[0:w, bg.shape[1] - w:bg.shape[1]] = bg[w:2*w, bg.shape[1] - w:bg.shape[1]]
    bg[bg.shape[0] - w:bg.shape[0], 0:w] = bg[bg.shape[0] - 2*w:bg.shape[0] - w, 0:w]
    bg[bg.shape[0]-w:bg.shape[0], bg.shape[1]-w:bg.shape[1]] = bg[bg.shape[0]-2*w:bg.shape[0]-w, bg.shape[1] - w:bg.shape[1]]
    return bg
