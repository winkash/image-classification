import argparse
import os
import numpy as np
import tempfile

from affine.model import Video, TrainingVideo, Label
from affine.detection.model.cross_validation import std_cross_validation

from affine.detection.vision.video_motioncolor.video_motioncolor_classifier \
    import VideoMotionColorClassifier, VideoInfoObject
from affine.detection.vision.video_motioncolor.inject_detector_to_db \
    import VideoMotionColorDetectorInjector
from affine.detection.utils import get_training_data as gtd

from logging import getLogger

logger = getLogger(__name__)


def download_video(v_name, v_id):
    """
    Downloads the video with v_id and stores it with the full path name provided
    in v_name if it's not already there. 

    Args:
        v_name: string, full path file name where to save the downloaded video
        v_id: long, video id of the video to be downloaded

    Returns:
        int, length of the downloaded video, in seconds.
        If the video file exists but the video_id is not on the DB, it returns -1
        If the file does not exist, and video_id is wrong or not on s3, returns 0

    Raises/Assertions:

    """
    video_length = -1

    v = Video.get(v_id)
    if os.path.exists(v_name):
        if v:
            video_length = v.length
    else:
        if v and v.s3_video:
            video_length = v.length
            logger.info("Downloading %d.flv ..." % (v.id))
            v.download_video(v_name)
        elif v:
            logger.error('Video is not on s3!')
            return 0
        else:
            logger.error('Video file does not exist!')
            return 0

    return video_length


def create_videoinfo_list(video_id_label_list, folder):
    """
    Creates a list of VideoInfoObject objects with the necessary information from
    each of the input video files.

    Args:
        video_id_label_list: 2-dimensional array that contains 
            in each row the basic information from one video as follows:
                [video_id video_label]
            If we don't know the labels, we can pass a list of video ids and they
            will automatically get label -1 as an output
        folder: string, 
            path to find (or store if they are not there) the video files 
            corresponding to each of the video_ids

    Returns:
        list_of_videos: list of VideoInfoObject objects. It contains one object 
            for each of the input video ids that could be used to find a corresponding video file
        list_of_labels: list with the labels corresponding to each video

    Raises/Assertions:

    """

    list_of_videos = []
    list_of_labels = []

    for v_info in video_id_label_list:
        if np.shape(v_info):
            lab = v_info[1]
            v_id = v_info[0]
        else:
            lab = -1
            v_id = v_info

        video_n = '%s/%d.flv' % (folder, v_id)
        frames_folder = '%s/%d_2fps_sc' % (folder, v_id)

        video_length = download_video(v_name=video_n, v_id=v_id)
        if video_length == 0:
            logger.info(
                'Video %s not found, skipping it for training' % (v_id))
        else:
            logger.info('Loaded video_id: %d \t label: %d'
                        % (v_id, lab))
            list_of_videos.append(VideoInfoObject(video_id=v_id,
                                                  videopath=video_n,
                                                  framespath=frames_folder,
                                                  length=video_length))
            if int(lab) > 0:
                list_of_labels.append(1)
            elif int(lab) == 0:
                list_of_labels.append(0)
            else:
                list_of_labels.append(-1)

    return list_of_videos, list_of_labels


def run_inject_classifier(
    model_dir, positive_samples_vids, target_label_id, detector_name
):
    """
    Creates VideogGame injector object, gets the positive training ids and
    injects the detector results

    Args:
        model_dir: path to folder with model files
        positive_samples_vids: list of video ids from positive training data
        target_label_id: long
        detector_name: string

    Returns:
        det, detector object created and injected on S3

    Raises/Assertions:

    """

    v_classifier = VideoMotionColorClassifier(
        os.path.join(model_dir, 'Configfile.cfg'))
    v_classifier.load_model(model_dir)

    di = VideoMotionColorDetectorInjector(v_classifier.model_dir)
    det = di.inject_detector(detector_name=detector_name,
                             label_id=target_label_id,
                             true_vid_list=positive_samples_vids,
                             confidence_th=v_classifier.confidence_th,
                             acceptance_th=v_classifier.accept_th)

    return det


def run_training_pipeline(
    video_id_label_list, input_dir, config_f, detector_name, target_label_id
):
    """ 
    Train and validate a motion-color based video classifier using the given training data,
    and create and inject the corresponding detector.

    This function runs the whole pipeline: sampling, feature extraction and\
     classifier cross-validation and training

    Args:
        video_id_label_list: 2-dimensional numpy array with video ids and the corresponding label.
            Each row contains: [ video_id, label]. 
            video_id_label_list[:,0] contains all the video ids
            video_id_label_list[:,1] contains all the corresponding labels

        input_dir: string, path to the folder where to find the training videos
            (or to download them if they are not there)

        config_f: string, path to find the config file (Configfile.cfg)

        detector_name: string. Note that the name of the detectors is unique, 
            they won't be over-written in the DB if they already exist

        target_label_id: int, label_id that we want to target with the detector
            e.g., 2418 -> IAB:Video & Computer Games (target)

    Returns:
        Boolean, indicates if the training was sucessful or not
            according to the average accuracy obtained from crossvalidation

    Raises/Assertions:
        AssertionError: get_config raises AssertionException if configfile \
            has bad formatting: 'Config file validation failed'
        Asserts if the list of videos has less than 8 videos
        Asserts if the number of videos is smaller than the number of folds \
            required for cross validation
    """

    assert (len(video_id_label_list) > 0 and
            len(video_id_label_list.shape) == 2), \
        "training info list can't be empty and has to be a 2D numpy array"

    logger.info('  Initialize Classifier and load Config ')
    model_dir = tempfile.mkdtemp()
    v_classifier = VideoMotionColorClassifier(configfile_name=config_f)

    n_folds = 1 / v_classifier.crossval_test_size
    unique_labels = set(video_id_label_list[:, 1])
    assert (len(unique_labels) >= 2), \
        "All the samples are from a unique class. We need at least 0 and 1"

    all_labels = video_id_label_list[:, 1]
    for l in unique_labels:
        num_videos_label_l = len(all_labels[all_labels == l])
        assert (num_videos_label_l >= n_folds), \
            "We need at least as many videos from each class as cross-validation folds"

    logger.info('  Get training data in the right format ')
    list_of_train_videos, list_of_labels = create_videoinfo_list(
        video_id_label_list=video_id_label_list, folder=input_dir)

    logger.info("  1 - Computing descriptors ")
    X, ok_video_indx = v_classifier.compute_descriptors(list_of_train_videos)
    Y = np.array([list_of_labels[i] for i in ok_video_indx])

    list_used_train_vids = np.array(
        [list_of_train_videos[i].video_id for i in ok_video_indx])
    list_used_labels = Y

    logger.info(" 2 - Computing cross validation ... ")
    scores = std_cross_validation(v_classifier, X, Y, n_folds=n_folds,
                                  average=None)
    v_classifier.prec = np.median(scores['precision'])
    logger.info(' \tCROSS VALIDATION median prec %f' % v_classifier.prec)

    det = None
    TARGET_LABEL = 0
    if v_classifier.prec >= v_classifier.crossval_th:
        logger.info(' 3 - Training with all data and saving model')
        v_classifier.train(X, Y)
        v_classifier.save_model(model_dir)
        logger.info('\t Model files saved in %s :' % (v_classifier.model_dir))

        logger.info(' 4 - Inject detector')
        di = VideoMotionColorDetectorInjector(v_classifier.model_dir)
        positive_samples_vids = \
            list_used_train_vids[list_used_labels == TARGET_LABEL]
        det = di.inject_detector(detector_name=detector_name,
                                 label_id=target_label_id,
                                 true_vid_list=positive_samples_vids,
                                 confidence_th=v_classifier.confidence_th,
                                 acceptance_th=v_classifier.accept_th)

        logger.info('\t Detector injected %s :' % (det))
        pos_train_vids_file = \
            os.path.join(v_classifier.model_dir, 'positive_train_vids_used')
        np.save(pos_train_vids_file, positive_samples_vids)
        logger.info('\t Positive training vids used stored in  %s :'
                    % (pos_train_vids_file))
    else:
        logger.error(
            'Cross validation results are not good enough to store this classifier')

    return v_classifier.model_dir, det


def run_get_training_data(
    target_label_id, npy_training_info_file=None, old_detector_id=None,
    max_num_pos_videos=500, ratio_neg_pos=5, excluded_labels_file=None
):
    """
    Append given training data (if any) with new available video ids from
    MTurk hits
    If no npy is provided, positive are obtained from TrainingVideo and
    negative from random labels

    Args:

        target_label_id: label id that we want to get training data for

        npy_training_info_file: file with already labeled video ids 
            (same format as output data in this function. column 0: video ids; column 1: label)

        old_detector_id: detector id from previous version of a similar detector, 
            if we want to use that detector training data to train this new one, 
            i.e., we will add as positive training data all the video_id entries 
                in TrainingVideo with this detector_id

        max_num_pos_videos: upper limit on the number of positive training data 
            (this function will try to get at least 6 times more negative data than positive, if it exists)

        RATIO_NEG_POS: ratio between negative and positive data 
            (e.g., ratio 10 means #neg = 10 * #pos)

        excluded_labels_file: npy file with a list of ids from labels 
            that we dont want to use as negative training data 
            (e.g., specific types of videogames for the videogame classifier)

    Returns:
        numpy array with:
        - first column containing video ids to be used for training
        - second column containing the corresponding label: 0 - target_label    according to MTurk results
                                                            1 - any other label

    Raises/Assertions:
        asserts if label_id given as target label does not exist in the DB
    """
    assert Label.get(target_label_id), "Target label does not exist"

    video_id_label_list = []
    positive_vids = []
    negative_vids = []
    new_positive_vids_from_tp = []
    new_negative_vids_from_fp = []

    if npy_training_info_file and os.path.exists(npy_training_info_file):
        list_vids_label = np.load(npy_training_info_file)
        all_vids = list_vids_label[:, 0]
        all_labels = list_vids_label[:, 1]
        positive_vids = list(all_vids[all_labels == 0])
        negative_vids = list(all_vids[all_labels > 0])
    elif old_detector_id:
        res = TrainingVideo.query.filter_by(
            detector_id=old_detector_id)
        positive_vids = [t.video_id for t in res]

    if max_num_pos_videos > len(positive_vids):
        new_positive_vids_from_tp = gtd.get_list_of_videoids(
            target_label_id=target_label_id, target_result=True,
            excluded_label_list=[],
            maxNumVideos=max_num_pos_videos - len(positive_vids)
        )

    new_negative_vids_from_fp = gtd.get_list_of_videoids(
        target_label_id=target_label_id, target_result=False,
        excluded_label_list=[], maxNumVideos=max_num_pos_videos
    )

    positive_vids = np.unique(positive_vids + new_positive_vids_from_tp)
    logger.info('Number of positive ids: %d' % len(positive_vids))

    negative_vids = np.unique(negative_vids + new_negative_vids_from_fp)
    exclusion_label_ids = []
    if len(negative_vids) < ratio_neg_pos * len(positive_vids):
        if excluded_labels_file:
            exclusion_label_ids = list(np.load(excluded_labels_file))
        random_negative_vids = gtd.get_list_of_videoids(
            target_label_id=None,
            excluded_label_list=[target_label_id] + exclusion_label_ids,
            maxNumVideos=len(positive_vids) *
            ratio_neg_pos - len(negative_vids)
        )
        negative_vids = np.unique(
            np.append(negative_vids, random_negative_vids))
    logger.info('Number of negative ids: %d' % len(negative_vids))

    video_id_label_list = np.zeros(
        (len(positive_vids) + len(negative_vids), 2))

    video_id_label_list[:, 0] = np.append(positive_vids, negative_vids)
    zeros_arr = np.zeros((len(positive_vids), 1))
    ones_arr = np.ones((len(negative_vids), 1))
    video_id_label_list[:, 1] = np.append(zeros_arr, ones_arr)

    return video_id_label_list


def main():
    """
    Run an example of the full pipeline to build a motion-color based video detector
    More details of each argument used in this main() in run_get_training_data and run_training_pipeline methods

    Example:
        python run_video_motioncolor_detection_pipeline.py \
            -input './testdata' -configfile './Configfile.cfg' \
            -labelInfoFile 'labeledVideosIds.npy' -detectorname 'videogames'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-input",
                        help="Path to folder with videos\
                        (or to download them if they are not there)")
    parser.add_argument("-configfile",
                        help="Path to the config file")
    parser.add_argument('-labelInfoFile',
                        help="Path to .npy file with video ids and labels")
    parser.add_argument('-excludedLabelInfo', default=None,
                        help="Path to .npy file with label ids that we dont "
                        "want as negative training")
    parser.add_argument('-detectorname',
                        help="Name for the detector to be created")
    parser.add_argument("-target_label_id", type=int,
                        help="Target label id")
    parser.add_argument("-max_positive_samples", type=int, default=500,
                        help="Max number of positive samples we want to use")
    parser.add_argument("-ratio_neg_pos", type=float, default=5.0,
                        help=" ratio_neg_pos * #positive data =  #negative_data")
    args = parser.parse_args()

    logger.info('\n-------- GET TRAINING DATA ------------')
    video_id_label_list = run_get_training_data(
        target_label_id=args.target_label_id,
        npy_training_info_file=args.labelInfoFile, old_detector_id=None,
        max_num_pos_videos=args.max_positive_samples,
        ratio_neg_pos=args.ratio_neg_pos,
        excluded_labels_file=args.excludedLabelInfo
    )

    logger.info('\n--------  BUILD CLASSIFIER and DETECTOR ------------')
    run_training_pipeline(
        video_id_label_list=video_id_label_list, input_dir=args.input,
        config_f=args.configfile, detector_name=args.detectorname,
        target_label_id=args.target_label_id)

if __name__ == "__main__":
    main()
