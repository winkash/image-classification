from logging import getLogger

from affine.detection.nlp.nec.detection import strip_control_characters
from affine.detection.url_classification.url_client import UrlClient
from affine.detection.url_classification.url_config import CFG_FILE, CFG_SPEC,\
    SKIP_DOMAINS
from affine.model.detection import UrlClassifier
from affine.model.classifier_models import UrlModel
from affine.detection.utils.validate_config import validate_config

logger = getLogger(__name__)


def process_page(page, classifiers):
    """ Runs URL classification on a page"""
    if page.domain in SKIP_DOMAINS:
        logger.info("Skipping URL classification on page %d due to domain" % page.id)
        return

    logger.info("Running URL classification on page %d" % page.id)
    matching_classifiers = set()
    for clf in classifiers:
        clf.grab_files()
        config_file = clf.local_path(CFG_FILE)
        config_obj = validate_config(config_file, CFG_SPEC)
        url_model = UrlModel.by_name(config_obj['model_name'])
        url_client = UrlClient(url_model.id)
        # We see urls with weird characters once in a while
        # Webpge remote_id is utf8 encoded string
        cleaned_url = strip_control_characters(page.remote_id.decode('utf-8'))
        pred = url_client.predict(cleaned_url)
        if pred:
            logger.info("URL true detection (page_id:%d, classifier:%s)" % (page.id, clf.name))
            clf.save_result(page.id)
            matching_classifiers.add(clf)
    classifiers_to_delete = set(classifiers) - matching_classifiers
    classifier_ids_to_delete = [classifier.id for classifier in classifiers_to_delete]
    UrlClassifier.delete_detector_results(page, classifier_ids_to_delete)
