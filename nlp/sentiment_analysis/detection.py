from logging import getLogger

from affine import config
from affine.model import Label, SentimentClassifier, TextDetectorResult, ClassifierTarget
from sa.sent_analysis.Lexicon import SentiLexicon
from sa.sent_analysis.Lexical_Classifier import LexicalClassifier

logger = getLogger(__name__)

MIN_CHARS_TO_PREDICT = 1000

def process_page(page, clfs):
    """ Runs Sentitiment Analysis classification on a page"""
    logger.info("Running SA detection on page {}".format(page.id))
    assert len(clfs) == 1, 'we currently support only one classifier'
    clf = clfs[0]
    SentimentClassifier.delete_detector_results(page, [clf.id])
    # API expects utf8 encoded text
    is_negative = text_has_negative_sentiment(page.title_and_text.encode('utf-8'))
    if is_negative:
        logger.info("Sentiment for page_id {} is negative".format(page.id))
        TextDetectorResult.log_result(page.id, clf.clf_target.id)


def text_has_negative_sentiment(text, threshold=.8):
    # Ignore pages with little text content
    if len(text) < MIN_CHARS_TO_PREDICT:
        return False

    lexicon = SentiLexicon()
    classifier = LexicalClassifier(lexicon)
    neg_score = classifier.classify(text , cumulative=False)[1][1]
    return neg_score > threshold

