import requests
import unicodedata
from logging import getLogger

from affine import config
from affine.model import Label, NamedEntityClassifier, TextDetectorResult, ClassifierTarget

logger = getLogger(__name__)


DBPEDIA_PREFIX = 'DBpedia:'
BLACK_LISTED_SURFACE_FORMS = ['youtube']
MAX_TEXT_LEN = 10000


def process_page(page, clfs):
    """ Runs Named Entity classification on a page"""
    logger.info("Running NEC detection on page %d" % page.id)
    # We only supprt one classifier currently
    assert len(clfs) == 1
    true_clf_targets = classify_title(page)
    clf = clfs[0]
    NamedEntityClassifier.delete_detector_results(page, [clf.id])
    for clf_target in true_clf_targets:
        logger.info("NEC true detection (page_id:%d, clf_target_id:%s)" %
                    (page.id, clf_target.id))
        TextDetectorResult.log_result(page.id, clf_target.id)


def classify_title(page):
    """
    Classfies the title of the page via the NamedEntityClassifier.

    Args:
        page: The page object that need to be classified.

    Returns:
        A list with all the DBpedia lables present in the title.
    """
    full_annotation = spotlight_annotate(page.title_and_text)

    entity_types = set()
    title_offset = len(page.title)

    for entity in full_annotation:
        if int(entity['offset']) >= title_offset:
            break

        if entity['surfaceForm'].lower() in BLACK_LISTED_SURFACE_FORMS:
            continue
        entity_types.update(set(entity['types']))
    canonical_entity_types = _get_canonical_types(entity_types)
    true_clf_targets = _get_matching_clf_targets(canonical_entity_types)
    return true_clf_targets


def _get_canonical_types(types):
    canonical_types = [name for name in types if name.startswith(DBPEDIA_PREFIX)]
    return canonical_types


def _get_matching_clf_targets(entity_types):
    if not entity_types:
        return []
    clf = NamedEntityClassifier.query.one()
    return ClassifierTarget.query.filter(ClassifierTarget.clf_id == clf.id).join(Label, ClassifierTarget.target_label_id == Label.id).filter(Label.name.in_(entity_types)).all()


def spotlight_annotate(text, confidence=0.5, support=20, timeout=30):
    """
    Annotates the text using spotlight.

    Args:
        text: The text that should be annotated.
        confidence, support: Internal spotlight parameters.
        timeout: Wait time before the server is considered timed out.

    Returns:
        A list of dics containing the results of the classification.

    Raises:
         Exception: The server did not reposnded with proper JSON.
    """

    # some unicode characters are problematic
    text = _preprocess_text(text)

    #  '' returns a 400 error
    if text.strip() == '':
        return []
    spotlight_address = config.get('spotlight_server.address') + "/annotate/"


    data = {'confidence': confidence, 'support': support, 'text': text}
    headers = {'accept': 'application/json'}
    response = requests.post(spotlight_address, data=data,
                             headers=headers, timeout=timeout)

    if response.status_code != requests.codes.ok:
        response.raise_for_status()

    results = response.json()

    if results is None:
        raise Exception("There Response does not contain proper JSON")

    if 'Resources' not in results:
        return []

    return [_clean_dic(result) for result in results['Resources']]


def _clean_dic(original):
    """
    Cleans the key names and return a proper Python list where needed.
    """
    clean = {}
    for key, value in original.items():
        if key.startswith('@'):
            clean_key = key[1:]
        else:
            clean_key = key

        if clean_key == 'types':
            clean_value = value.split(',')
        else:
            clean_value = value

        clean[clean_key] = clean_value

    return clean


def _preprocess_text(text):
    """Trims overly lengthy text and removes characters that cause
    problems unicode problems in XML"""
    if len(text) > MAX_TEXT_LEN:
        text = text[:text.find(' ', MAX_TEXT_LEN)]
    return strip_control_characters(text)


def strip_control_characters(text):
    return "".join(ch for ch in text if unicodedata.category(ch)[0]!="C")
