import os

from collections import defaultdict

from affine.detection.vision.vision_text_detect.vision_text_detector import \
    VisionTextDetector
from affine.video_processing import image_path_to_time
from affine.model import Box, Label, Video, BoxDetectorResult,\
    TextDetectClassifier, ClassifierTarget


def save_boxes(l_id, video_id, boxes):
    """ Save boxes to TextBox table and detector result to BoxDetectorResult
    Args:
        l_id: target Label id
        video_id: Video id
        boxes: dict with key: timestamp, value: list of rectangles

    Returns:
        List of newly created box ids

    Raises/Assertions:
        Asserts if video id does not exist
        Asserts of label id does not exist
    """
    vid = Video.get(video_id)
    l = Label.get(l_id)
    assert vid, "Video %s does not exist" % video_id
    assert l, "Label %s does not exist" % l_id
    clf_target = ClassifierTarget.query.filter_by(target_label_id=l.id)\
        .join(TextDetectClassifier).one()
    box_ids = []
    for timestamp in boxes.keys():
        for h, w, y, x in boxes[timestamp]:
            box_id = Box.get_or_create(
                x=x, y=y, width=w, height=h, video=vid,
                timestamp=timestamp, box_type='Text')
            box_ids.append(box_id)
            BoxDetectorResult.log_result(box_id, clf_target.id)
    return box_ids


def judge_video(imagedir, model_dir, word_det_th):
    """
    Evaluate if the video contains text.

    Args:
        image_dir: string, full path to video frames
        model_dir: string, path to folder with classifier model files
        word_det_th: acceptance threshold from the config of
            VisionTextDetector

    Returns:
        bounding_rects_dict: dict with key: timestamp, value: list of
        bounding box info

    Raises/Assertions:
        Asserts if the imagedir passed is not there
    """
    assert (os.path.exists(imagedir)), \
        "Image dir %s does not exist" % (imagedir)

    config_file = os.path.join(model_dir, 'Configfile.cfg')
    vt_detector = VisionTextDetector(config_file)
    vt_detector.load_model(model_dir)
    if word_det_th >= .5 and word_det_th <= 1:
        vt_detector.pred_thresh = word_det_th

    img_files, bounding_rects = vt_detector.detect_text_in_video(
        imagedir)

    bounding_rects_dict = defaultdict(list)
    for br in bounding_rects:
        idx, h, w, x, y = br
        path = img_files[idx]
        path = os.path.abspath(path)
        timestamp = image_path_to_time(path)
        bounding_rects_dict[timestamp].append((h, w, x, y))

    return bounding_rects_dict
