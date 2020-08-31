import random
from datetime import datetime, timedelta
from logging import getLogger

import numpy as np
from affine.detection.vision.facerec import *
from affine.detection.model.features import RawPatch
from affine.detection.model.databag import DataBag
from affine.model import *

logger = getLogger('affine.data_acquisition')


def get_data(label_ids, num_boxes, ratio=0.8, min_boxes=40,
             min_confidence=0.0, min_width=50, single_box_per_video=True):
    """This method gets box data for training/testing on faces,
    Args:
        label_ids : can take the string 'all' meaning that all faces having
        True results will be returned OR
        num_boxes : max number of boxes you want for each label
        ratio : ratio in which data should be training into train/test
                can take a list of label_ids for which data is requested
        min_boxes : Will not include data of a label if unique videos found
                    is less than this value
        min_confidence : Include faces only having FaceInfo.confidence greater
                        than min_confidence
        min_width : Include faces having width > min_width
        single_box_per_video : If True will return only one box per video
    Returns:
        train_bag and test_bag, 2 data bags
    Raises:
        TypeError : if label_ids is not 'all' or a list(label_ids)
    """
    face_signature_client = DynamoFaceSignatureClient()
    train_bag = DataBag([], [], [])
    test_bag = DataBag([], [], [])

    if label_ids == 'all':
        label_ids = {l for (l,) in session.query(MTurkBox.label_id.distinct())}
    elif not isinstance(label_ids, list):
        raise TypeError("label_ids should be 'all' or a list of label_ids")

    for label_id in label_ids:
        logger.info('Getting data for %s' % label_id)

        query = session.query(MTurkBox.box_id, Box.video_id).\
                filter(MTurkBox.box_id == Box.id)
        query = query.filter(MTurkBox.label_id == label_id,
                MTurkBox.result == True, Box.width >= min_width)
        query = query.join(Box.face_info).\
                filter(FaceInfo.confidence > min_confidence)
        if single_box_per_video:
            query = query.group_by(Box.video_id)
        limit = 0
        boxes = []
        for box_id, video_id in query:
            if limit < num_boxes and \
                    face_signature_client.has_signature(box_id):
                limit += 1
                boxes.append(box_id)

        logger.info('Got %d boxes for %s' % (len(boxes), label_id))
        random.shuffle(boxes)
        count = int(len(boxes) * ratio)
        test_count = len(boxes) - count

        if count < min_boxes:
            logger.info("Got %d (< min_boxes %d), will not include"
                    % (count, min_boxes))
            continue

        train_bag.box_ids += boxes[:count]
        train_bag.feats += face_signature_client.get_signatures(boxes[:count])
        train_bag.labs += [label_id] * count

        test_bag.box_ids += boxes[count:]
        test_bag.feats += face_signature_client.get_signatures(boxes[count:])
        test_bag.labs += [label_id] * test_count

    train_bag.feats = np.asarray(train_bag.feats)
    test_bag.feats = np.asarray(test_bag.feats)

    logger.info("Got total %d training boxes" % len(train_bag.box_ids))
    logger.info("Got total %d testing boxes" % len(test_bag.box_ids))

    return train_bag, test_bag


def get_negative_data(count, ratio=0.8, min_confidence=0.0, min_width=50,
        single_box_per_video=True):
    """This method gets negative box data for training/testing,
    Args:
        count : max number of boxes you want
        ratio : ratio in which data should be divided into
        min_confidence : Include faces only having FaceInfo.confidence
        greater than min_confidence
        min_width : Include faces having width > min_width
        single_box_per_video : If True will return only one box per video
    Returns:
        train_bag and test_bag, 2 data bags
    """
    face_signature_client = DynamoFaceSignatureClient()
    neg_train = DataBag([], [], [])
    neg_test = DataBag([], [], [])

    video_ids = set()
    neg_boxes = set()
    logger.info('Starting to acquire Negative Data')
    while len(neg_boxes) < count:
        qneg = session.query(Box.id, Box.video_id).\
                filter(Box.width >= min_width).filter(Box.box_type=='Face')
        if video_ids:
            qneg = qneg.filter(~Box.video_id.in_(video_ids))
        qneg = qneg.outerjoin(MTurkBox, MTurkBox.box_id == Box.id).\
                filter(MTurkBox.result != True)
        qneg = qneg.join(FaceInfo, FaceInfo.box_id == Box.id).\
                filter(FaceInfo.confidence >= min_confidence)
        limit = 0
        new_boxes = set()
        if single_box_per_video:
            for (box_id, video_id) in qneg:
                if limit < count and video_id not in video_ids \
                        and face_signature_client.has_signature(box_id):
                    limit += 1
                    new_boxes.add(box_id)
                    video_ids.add(video_id)
        else:
            for (box_id, video_id) in qneg:
                if limit < count \
                        and face_signature_client.has_signature(box_id):
                    limit += 1
                    new_boxes.add(box_id)
                    video_ids.add(video_id)
        if new_boxes:
            neg_boxes.update(new_boxes)
            logger.info("Total boxes acrued so far : %s" % len(neg_boxes))
        else:
            logger.warn('Did not find any more negative boxes, Exiting')
            break

    if not len(neg_boxes):
        raise ValueError("no Negative boxes not found")

    neg_boxes = list(neg_boxes)
    random.shuffle(neg_boxes)
    filter_neg_boxes = neg_boxes[:count]

    # Extracting features for negative data
    neg_feats = face_signature_client.get_signatures(filter_neg_boxes)

    train_count = int(len(filter_neg_boxes) * ratio)
    neg_train.box_ids = filter_neg_boxes[:train_count]
    neg_train.feats = neg_feats[:train_count]
    neg_train.labs = [-1] * train_count
    neg_test.box_ids = filter_neg_boxes[train_count:]
    neg_test.feats = neg_feats[train_count:]
    neg_test.labs = [-1] * len(neg_test.box_ids)

    logger.info('Got total negative training boxes : %s'
            % len(neg_train.box_ids))
    logger.info('Got total negative testing boxes : %s'
            % len(neg_test.box_ids))

    return neg_train, neg_test

def get_data_from_domains(domains, min_confidence=0.0, min_width=50,
        min_last_crawled_days=150, limit=None):
    """
    Args:
        domains : List of domains from which boxes are to be collected
        min_confidence : boxes only having confidence above minimum confidence
        will be picked up
        min_width : boxes only having width greater than minimum width will
        be picked up
        min_last_crawled_days : only boxes which belong to urls crawled in
        the last min_last_crawled_days days
        limit : number of boxes that should be returned
    Returns:
        databag, containing all the boxes and corresponding signatures
    """
    min_last_crawled = datetime.utcnow() - \
            timedelta(days=min_last_crawled_days)
    query = session.query(Box.id.distinct()).filter(Box.width >= min_width).\
            filter(Box.box_type=='FACE').\
            join(Box.face_info).filter(FaceInfo.confidence > min_confidence)
    query = query.outerjoin(MTurkBox, MTurkBox.box_id == Box.id).\
            filter((MTurkBox.result != True) | (MTurkBox.result == None))
    query = query.join(VideoOnPage, Box.video_id == VideoOnPage.video_id).\
            filter_by(active=True, is_preroll=False)
    query = query.join(WebPage, WebPage.id == VideoOnPage.page_id).\
            filter(WebPage.domain.in_(domains),
                   WebPage.last_crawled_video > min_last_crawled)
    box_ids = [box_id for (box_id,) in query.order_by(WebPage.\
            last_crawled_video.desc()).limit(limit)]

    if box_ids:
        feats = get_or_extract_signatures(box_ids)
        labels = [-1] * len(box_ids)
        bag = DataBag(feats, labels, box_ids)
        logger.info("Got total %d boxes from given domains", len(bag.box_ids))
        return bag
    else:
        raise ValueError("No boxes found")

def get_or_extract_signatures(box_ids):
    face_signature_client = DynamoFaceSignatureClient()
    feats_dict = dict(zip(
                box_ids,
                face_signature_client.get_signatures(box_ids, False))
            )
    new_feats_dict = {}

    boxes_missing_signatures = [box for box, feat in feats_dict.iteritems()
                                if feat is None]
    if boxes_missing_signatures:
        rp = RawPatch()
        feats = rp.extract(boxes_missing_signatures)
        new_feats_dict = dict(zip(boxes_missing_signatures, feats.tolist()))
    feats_dict.update(new_feats_dict)
    feats = []

    for box_id in box_ids:
        if box_id in feats_dict.keys():
            assert feats_dict[box_id] is not None, \
            "Could not extract signature for box_id : %s" % box_id
            feats.append(feats_dict[box_id])
        else:
            raise ValueError("Missing signature for box : %s" % box_id)
    return feats
