import os

from affine.detection.vision.vision_text_recognize.vision_text_recognizer import \
    VisionTextRecognizer
from affine.model import TextBoxResult, session, Box


def save_results(boxes_dict):
    """
    Save words to text_box_results table
    Args:
        boxes: dict of box ids to words (strings)
    """
    for b_id in boxes_dict:
        if len(boxes_dict[b_id]):
            TextBoxResult.log_result(b_id, boxes_dict[b_id])


def judge_video(video_id, imagedir, model_dir, rec_th):
    """
    Evaluate if the boxes contain words.

    Args:
        video_id: video id
        imagedir: string path to images
        model_dir: string, path to folder with classifier model files
        word_det_th: acceptance threshold from the config of
            VisionTextRecognizer

    Returns:
        box_dict: dict with key: box id, value: string

    Raises/Assertions:
        Asserts if the imagedir passed does not exists
    """
    assert (os.path.exists(imagedir)), \
        "Image dir %s does not exist" % (imagedir)

    boxes = session.query(Box.id).filter(
        Box.video_id == video_id, Box.box_type == 'Text')
    boxes = [b_id for b_id, in boxes]
    config_file = os.path.join(model_dir, 'Configfile.cfg')
    vt_recognizer = VisionTextRecognizer(config_file)
    vt_recognizer.load_model(model_dir)
    if rec_th >= 0 and rec_th <= 1:
        vt_recognizer.pred_thresh = rec_th

    box_dict = vt_recognizer.recognize_text_in_video(imagedir, boxes)
    return box_dict
