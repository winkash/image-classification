import random
import tempfile
from sqlalchemy.sql.expression import func
from affine.model import Video
from affine.detection.model.features import BagOfWords
from affine.detection.vision.utils.scene_functions import download_images_to_dir


def train_bag_of_words(vocabsize, fts_ex, logo_paths, num_images=1000):
    """Trains a bag of words from random images and logo images

    Logo images are given in the input and random images are downloaded
    from s3 video samples

    Args:
        vocabsize: size of the vocabulary
        fts_ex: The feature extraction method for learning the bag of words
        logo_paths: paths to the logo images
        num_images : number of random images

    returns:
        a tuple of the learned bag of words and path to all images
    """

    imgs = grab_from_video(num_images)
    download_dir = tempfile.mkdtemp()
    rand_imgs_paths = download_images_to_dir(imgs, download_dir)
    ims_paths = rand_imgs_paths + logo_paths
    bow = trained_bow_grabber(vocabsize, fts_ex, ims_paths)
    return (bow, ims_paths)


def grab_from_video(num_images):
    """ Grabs sample s3 images from videos

    Args:
        num_images: number of random images

    Returns:
        list of the random images that has been grabbed
    """
    images = []
    rand_int = random.randint(31, 59)
    query = Video.query.filter(Video.id % rand_int == 0, Video.s3_video == True)
    query = query.order_by(Video.added.desc())
    for v in query.limit(num_images*2).yield_per(10):
        length = len(v.s3_timestamps()) / 2
        if length:
            ts = v.s3_timestamps()[length]
            images.append((v.id, ts))
        if len(images) >= num_images:
            break
    return images[:num_images]


def trained_bow_grabber(vocabsize, fts_ex, ims_paths):
    """ Trains a bag of words from the given images

    Args:
        vocabsize: size of the vocabulary
        fts_ex: The feature extraction method for learning the bag of words
        ims_paths: paths to the images

    Returns:
        a trained bag of words object
    """

    bow = BagOfWords(fts_ex, vocabsize)
    bow.train(ims_paths)
    return bow
