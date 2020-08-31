import os
import cv2
from collections import defaultdict
from tempfile import mkstemp

from affine.model import Label, Video, ResultAggregator
from .face_extract import FaceExtractor
from .injection import FaceRecognizeClassifierInjector
from affine.detection.model_worker.tasks.data_processor_task import DataProcessorClient,\
    convert_files_to_bin
from affine.video_processing import sample_images, image_path_to_time,\
    time_to_image_path

FACE_IMAGES_PER_VIDEO = 20
FACE_MIN_IMAGES_PER_VIDEO = 0
MIN_OCCURENCE_FOR_VIDEO = 2
FACE_MIN_CONFIDENCE = 0.0
FACE_CELERY_TIMEOUT = 30

def detect_judge_video(imagedir):
    face_images = sample_images(imagedir, FACE_IMAGES_PER_VIDEO, FACE_MIN_IMAGES_PER_VIDEO)
    extractor = FaceExtractor()
    ra = ResultAggregator()
    for image in face_images or []:
        ts = image_path_to_time(image)
        face_boxes = extractor.extract_faces(image)
        for x, y, w, h in face_boxes:
            ra.add_new_box(x, y, w, h, ts, 'Face')
    return ra.result_dict


def recognize_judge_video(clf_dir, video_id, imagedir):
    model_name = FaceRecognizeClassifierInjector.get_model_name(clf_dir)
    dp_client = DataProcessorClient(model_name)

    results = {}
    ra = ResultAggregator()
    votes = defaultdict(int)
    video = Video.get(video_id)
    assert video
    for box in video.face_boxes:
        path = time_to_image_path(imagedir, box.timestamp)
        fd, cropped_path = mkstemp(suffix='.jpg')
        os.close(fd)
        try:
            rect = get_rect_to_recognize(box)
            crop_image(path, cropped_path, *rect)
            [bin_data] = convert_files_to_bin([cropped_path])
            result = dp_client.predict(bin_data, box.width, box.height, async=True)
            results[box.id] = result
        finally:
            os.remove(cropped_path)
    for box_id, result in results.iteritems():
        label_id, conf, parts = result.wait(timeout=FACE_CELERY_TIMEOUT)
        if conf is not None:
            ra.add_face_info(box_id, conf, parts)
            if conf > FACE_MIN_CONFIDENCE and label_id is not None:
                assert Label.get(label_id)
                ra.add_box_result(box_id, label_id)
                votes[label_id] += 1
    for label_id, occur in votes.iteritems():
        if occur >= MIN_OCCURENCE_FOR_VIDEO:
            ra.add_video_result(label_id)
    return ra.result_dict


def get_rect_to_recognize(box):
    # Because of the binary used for face detection, boxes
    # may have negative x, y coordinates.
    return max(box.x, 0), max(box.y, 0), box.width, box.height


def crop_image(path, cropped_path, x, y, w, h):
    img = cv2.imread(path)
    crp = img[y:y+h, x:x+w]
    assert cv2.imwrite(cropped_path, crp)
