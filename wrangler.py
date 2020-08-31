"""module for controlling the running of detectors"""
from logging import getLogger

from sqlalchemy import or_

from affine import librato_tools
from affine.video_processing import sample_images, time_to_image_path
from affine.detection.nlp.lda import detection as lda_detection
from affine.detection.nlp import language_detection
from affine.detection.nlp.ner import detection as ner_detection
from affine.detection.nlp.nec import detection as nec_detection
from affine.detection.url_classification import detection as url_detection
from affine.detection.nlp.sentiment_analysis import detection as sa_detection
from affine.detection.text_extractor import TextExtractor
from affine.model import Settings, LanguageDetector, LdaDetector, NerDetector,\
    TextDetectionVersion, TextDetectorResult, session,\
    UrlClassifier, ClassifierTarget, flush_detector_log_to_db,\
    NamedEntityClassifier, SentimentClassifier

logger = getLogger(__name__)

ENGLISH_LABEL_NAME = 'English'


def process_asr(video, video_path):
    """ Run ASR on video """
    if not video.s3_transcript:
        t = TextExtractor(video, video_path)
        logger.info('Transcribing speech to text for video: %s ' % video.id)
        t.transcribe_video()
        logger.info(
            'Done transcribing speech to text for video: %s ' % video.id)


def determine_run_order(dets):
    return sorted(dets, lambda d1, d2: cmp(d1.run_group, d2.run_group))


@librato_tools.timeit('video-processor.process_video')
def process_video(video, video_path, imagedir, detectors):
    """process_video runs detectors on a given video and saves the results for
    each detector.

    @params:
        video      - The video object to process.
        video_path - path to the video file.
        imagedir   - path to images for video
        detectors  - list of detectors who we want to insure have results for
                     given video
    """
    for detector in determine_run_order(detectors):
        logger.info('Detecting %s, %s', detector, video.id)
        detector.process_video(video.id, video_path, imagedir)

    if Settings.get_value('save_transcript'):
        process_asr(video, video_path)


# only report times > 1 second
@librato_tools.timeit('text-processor.process_text', lambda x: x > 1)
def process_text(page, tm_classifier):
    lang_label = process_language(page)
    if lang_label.name == ENGLISH_LABEL_NAME:
        process_url(page)
        process_tm(page, tm_classifier)
        process_lda(page)
        process_ner(page)
        process_nec(page)
        process_sa(page)
    else:
        logger.info(
            'Skipping topic modeling for Non-English page %s' % page.id)


@librato_tools.timeit('text-processor.process_language', lambda x: x > 1)
def process_language(page):
    det = LanguageDetector.query.one()
    if page.text_detection_update is None or page.text_detection_update <= det.updated_at:
        lang_label = language_detection.process_page(page)
    else:
        res = session.query(TextDetectorResult, ClassifierTarget).\
            filter_by(page_id=page.id).\
            filter(ClassifierTarget.id == TextDetectorResult.clf_target_id).\
            filter(ClassifierTarget.clf_id == det.id).\
            all()
        # if returns a list of multiple languages or empty list for some
        # reason, we rerun language detection
        if len(res) != 1:
            lang_label = language_detection.process_page(page)
        else:
            [(tdr, clf_target)] = res
            lang_label = clf_target.target_label
    return lang_label


@librato_tools.timeit('text-processor.process_tm', lambda x: x > 1)
def process_tm(page, tm_classifier):
    tm_timestamp = TextDetectionVersion.get_latest_version_timestamp(
        detector_type='topic_model')
    if page.text_detection_update is None or page.text_detection_update <= tm_timestamp:
        tm_classifier.process_page(page)


def process_text_classifier(detector_cls, detection_module, page):
    query = detector_cls.query.filter(
        detector_cls.enabled_since != None,
    )
    if page.text_detection_update is not None:
        query = query.filter(or_(
            detector_cls.enabled_since >= page.text_detection_update,
            detector_cls.updated_at >= page.text_detection_update
        ))
    detectors = query.all()
    if detectors:
        detection_module.process_page(page, detectors)


@librato_tools.timeit('text-processor.process_lda', lambda x: x > 1)
def process_lda(page):
    process_text_classifier(LdaDetector, lda_detection, page)


@librato_tools.timeit('text-processor.process_ner', lambda x: x > 1)
def process_ner(page):
    process_text_classifier(NerDetector, ner_detection, page)


@librato_tools.timeit('text-processor.process_nec', lambda x: x > 1)
def process_nec(page):
    process_text_classifier(NamedEntityClassifier, nec_detection, page)


@librato_tools.timeit('text-processor.process_url', lambda x: x > 1)
def process_url(page):
    process_text_classifier(UrlClassifier, url_detection, page)

@librato_tools.timeit('text-processor.process_sa', lambda x: x > 1)
def process_sa(page):
    process_text_classifier(SentimentClassifier, sa_detection, page)

