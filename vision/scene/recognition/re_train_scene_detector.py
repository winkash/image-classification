#!/usr/bin/env python
import os
import argparse
import random
import sys

from configobj import ConfigObj
from logging import getLogger

from affine import config as affine_config
from affine.aws import s3client
from affine.model import Video, SceneDetector, MTurkImageJob, MTurkImage, \
         ImageHit, ImageDetectorResult, session, ClassifierTarget
from affine.detection.vision.scene.recognition.scene_classifier import SceneClassifier
from affine.detection.vision.scene.recognition.scene_dataset import SceneDataset


logger = getLogger(__name__)

MAX_POS = 1500
MAX_NEG = 5000


def generate_info_files(s3_info_filename, info_filename, pos_dir, neg_dir):
    "generate info files for training and testing"
    if not os.path.exists(pos_dir):
        os.makedirs(pos_dir)
    if not os.path.exists(neg_dir):
        os.makedirs(neg_dir)
    bucket = affine_config.s3_bucket()
    fp = open(s3_info_filename, 'r')
    fo = open(info_filename, 'w')
    for f in fp:
        line = f.split(' ')
        image_label = int(line[1])
        video_id = int(line[2])
        time_stamp = int(line[3])
        filename = '%012d_%012d.jpg' % (video_id, time_stamp)
        video = Video.get(video_id)
        if video:
            all_tmps = video.s3_timestamps()
            if time_stamp in all_tmps:
                if image_label > 0:
                    outfile = os.path.join(pos_dir, filename)
                else:
                    outfile = os.path.join(neg_dir, filename)
                line_item = '%s %i %i %i' % (outfile, image_label, video_id, time_stamp)
                if os.path.exists(outfile):
                    fo.write(line_item + '\n')
                    continue
                print "downloading data from s3"
                # using the affine bucket for negative images
                img_path = 'thumbnail/%d/%d' % (video_id, time_stamp)
                s3client.download_from_s3(bucket, img_path, outfile)
                fo.write(line_item + '\n')
    fp.close()
    fo.close()


def split_set(image_set, split_ratio):
    """split image set given the ratio for train/test.
    It splits the sets wrt the number of total frames.
    We leave the frames from the same video in the same set.
    """
    index = []
    videos = []
    mypath = []
    for path, label, vid_id, ts in image_set:
        index.append((vid_id, ts))
        mypath.append((path,label,vid_id,ts))
        videos.append(vid_id)

    index = list(set(index))
    index.sort()
    sorted_videos = list(set(videos))
    sorted_videos.sort()
    mypath =  list(set(mypath))
    mypath = sorted(mypath, key=lambda x: x[2])
    n_frames = len(index)
    optimal_n_frames_train = int(n_frames * split_ratio)

    if optimal_n_frames_train > 0:
        last_pair = index[optimal_n_frames_train - 1]
        rest = index[optimal_n_frames_train:]
        from_same_video = [item for item in rest if item[0] == last_pair[0]]
    else:
        from_same_video = []
    n_frames_in_train = optimal_n_frames_train + len(from_same_video)
    train_set = []
    test_set = []
    #case where test set is empty because we pull all the frames in one set
    if n_frames_in_train == n_frames: #divide by video
        nvid = len(sorted_videos)
        ntrain = int(nvid * split_ratio) #number of videos in training set
        for i in xrange(ntrain):
            vv = sorted_videos[i]
            for p in mypath:
                if p[2] == vv:
                    train_set.append(p)

        for i in xrange(ntrain,nvid):
            vv = sorted_videos[i]
            for p in mypath:
                if p[2] == vv:
                    test_set.append(p)
    else:
        for i in xrange(n_frames_in_train):
            train_set.append(mypath[i])

        for i in xrange(n_frames_in_train, n_frames):
            test_set.append(mypath[i])
    return train_set, test_set


def get_set_lines(bucket, min_neg_lines_to_use, train_pos, inter_dir, s3_filename):
    if train_pos < min_neg_lines_to_use:
        n_neg_lines_to_use = min_neg_lines_to_use
    else:
        n_neg_lines_to_use = train_pos
    local_filename = os.path.join(inter_dir, s3_filename)
    s3client.download_from_s3(bucket, s3_filename, local_filename)
    with open(local_filename, 'r') as fo:
        all_neg = fo.read().splitlines()
    neg = []
    t_set = random.sample(xrange(1, len(all_neg)), n_neg_lines_to_use)
    for i in t_set:
        s3, lb, v, tm =  all_neg[i].split()
        if Video.get(v):
            neg.append(all_neg[i])
    return neg


def get_negative_set_s3(config, inter_dir, train_pos, test_pos):
    ''' Predetermined set of negative images from s3 '''
    bucket = affine_config.s3_detector_bucket()  # CHANGE
    min_neg_lines_to_use = int(config['train_detector_params']['neg_train_min_num_frames'])
    neg_train = get_set_lines(bucket, min_neg_lines_to_use, train_pos, inter_dir, 's3_neg_test_info_readymade.txt')
    min_neg_lines_to_use = int(config['train_detector_params']['neg_test_min_num_frames'])
    neg_test = get_set_lines(bucket, min_neg_lines_to_use, test_pos, inter_dir, 's3_neg_train_info_readymade.txt')
    return neg_train, neg_test


def is_original_scene(det_id):
   det = SceneDetector.query.filter_by(id = det_id).first()
   imjob = MTurkImageJob.query.filter_by(label_id = det.clf_target.target_label_id).first()
   if imjob:
       return True
   else:
       return False


def get_positive_training(det, ratio):
    """ Get positive images obtained during training using mturk"""
    images = session.query(MTurkImage).join(ImageHit).join(MTurkImageJob)
    images = images.filter(MTurkImage.image_hit_id == ImageHit.id,
                           MTurkImage.result == True,
                           ImageHit.mturk_image_job_id == MTurkImageJob.id,
                           MTurkImageJob.label_id == det.clf_target.target_label_id)
    images = images.limit(MAX_POS)
    image_count = images.count()
    if not image_count:
        raise Exception('No positive examples found in mturk for detector %s' % det.name)
    pos_image_set = []
    for mi in images:
        s3_path = Video.construct_s3_image_url(mi.video_id, mi.timestamp)
        pos_image_set.append([s3_path, 1, mi.video_id, mi.timestamp])
    pos_train, pos_test = split_set(pos_image_set, ratio)
    return pos_train, pos_test


def get_false_positives(det, ratio, config2, inter_dir):
    """ Get negative images obtained during QA using mturk (false positives)"""
    clf_target = det.clf_target
    im = session.query(MTurkImage).filter_by(label_id=clf_target.target_label_id,
                                             result=False)
    im = im.join(ImageDetectorResult,
                 MTurkImage.video_id == ImageDetectorResult.video_id and
                 MTurkImage.timestamp == ImageDetectorResult.time)
    im = im.filter(ImageDetectorResult.clf_target_id == det.clf_target.id).\
        join(ImageHit, MTurkImage.image_hit_id == ImageHit.id).\
        filter(ImageHit.mturk_image_job_id==None)

    images = [(t.video_id, t.timestamp) for t in im]
    neg_image_set = []
    imgcount = len(images)
    if imgcount < MAX_NEG:
        config2['train_detector_params']['neg_train_min_num_frames'] = 0
        other, _ = get_negative_set_s3(config2, inter_dir, MAX_NEG-imgcount, 1)
        extra = []
        for s3_line in other:
            s3, lb, v, d = s3_line.split()
            extra.append((int(v),int(d)))
        images = images + extra

    if not images:
        raise Exception('No negative examples found in mturk for detector %s' % det.name)
    for vid, tm in images:
        s3_path = Video.construct_s3_image_url(vid, tm)
        neg_image_set.append([s3_path, -1, vid, tm])
    neg_train, neg_test = split_set(neg_image_set, ratio)
    return neg_train, neg_test

def get_s3_files(det_id, ratio, config2, inter_dir):
    """ Get S3 image paths into files and separate train/test sets """
    test = os.path.join(inter_dir, 's3_test_info.txt')
    train = os.path.join(inter_dir, 's3_train_info.txt')
    det = SceneDetector.get(det_id)
    test_data = []
    if not os.path.exists(test) or not os.path.exists(train):
        neg_train, neg_test = get_false_positives(det, ratio, config2, inter_dir)
        pos_train, pos_test = get_positive_training(det, ratio)
        if pos_test and neg_test:
            with open(test, 'w') as fo:
                for s3, la, vid, ts in pos_test:
                    fo.write('%s %i %i %i \n' % (s3, la, vid, ts))
                    test_data.append([vid, ts])
                for s3, la, vid, ts  in neg_test:
                    fo.write('%s %i %i %i \n' % (s3, la, vid, ts))
        if pos_train and neg_train:
            with open(train, 'w') as fo:
                for s3, la, vid, ts in pos_train:
                    fo.write('%s %i %i %i \n' % (s3, la, vid, ts))
                for s3, la, vid, ts in neg_train:
                    fo.write('%s %i %i %i \n' % (s3, la, vid, ts))
    else:
        logger.info("file already exists")
    num = [len(pos_train), len(pos_test), len(neg_train), len(neg_test)]
    return train, test, num, test_data

def get_s3_data(inter_dir, s3_info_filename_train, s3_info_filename_test):
    """ Generate info files and download images from S3 to local folder """
    train_info_file = os.path.join(inter_dir, 'scene_train_info.txt')
    pos_dir = os.path.join(inter_dir, 'train', 'pos_images')
    neg_dir = os.path.join(inter_dir, 'train', 'neg_images')
    generate_info_files(s3_info_filename_train, train_info_file, pos_dir, neg_dir)
    test_info_file = os.path.join(inter_dir, 'scene_test_info.txt')
    pos_dir = os.path.join(inter_dir, 'test', 'pos_images')
    neg_dir = os.path.join(inter_dir, 'test', 'neg_images')
    generate_info_files(s3_info_filename_test, test_info_file, pos_dir, neg_dir)
    return train_info_file, test_info_file


def re_train_scene_detector(config_filename, inter_dir):
    """ Main function to retrain a scene detector """
    config2 = ConfigObj(config_filename)
    det_id = config2['model_to_db_injection']['detector_id']
    ratio = float(config2['train_detector_params']['split_ratio'])
    logger.info("checking classifier")
    if not is_original_scene(det_id):
        raise IOError("Cannot retrain detector as classifier was not originally trained (it is a retrained version). Please choose the original scene classifier")
    logger.info("getting data from DB")
    train, test, numimg, pos_test = get_s3_files(det_id, ratio, config2, inter_dir)
    logger.info("downloading data from s3")
    train_info_file, test_info_file =  get_s3_data(inter_dir, train, test)
    #create model folder
    logger.info("re-training model")
    scene = SceneClassifier(config_filename, inter_dir)
    scene.set_params()
    scene.train_infofile = train_info_file
    scene.test_info_file = test_info_file
    scene.dataset = SceneDataset(scene.final_dir, scene.labels)
    scene.dataset.dataset['pos'].num_train = numimg[0]
    scene.dataset.dataset['pos'].num_test = numimg[1]
    scene.dataset.dataset['neg'].num_train = numimg[2]
    scene.dataset.dataset['neg'].num_test = numimg[3]
    scene.dataset.dataset['pos'].dsets['test'] = pos_test
    scene.train_classifier()
    results = scene.test_classifier()
    scene.evaluate_classifier(results)
    if scene.tpr >= TPR and scene.fpr <= FPR:
        scene.inject_classifier()
        print scene
    else:
       logger.info('Retrain classifier. TPR = %d , FPR = %d' % (scene.tpr, scene.fpr))
    logger.info("DONE")


if __name__ == '__main__':
    # parsing the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-inter_dir', action='store', dest='inter_dir', required=True, help='set the inter_dir trained detector')
    parser.add_argument('-config_file', action='store', dest='config_filename', required=True, help='set the config filename')
    results = parser.parse_args()
    if not os.path.exists(results.inter_dir):
        print '-f inter_dir  not found!'
        sys.exit(1)

    if not os.path.exists(results.config_filename):
        print '-config_file config filename  not found!'
        sys.exit(1)

    re_train_scene_detector(results.config_filename, results.inter_dir)
