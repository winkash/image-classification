from logging import getLogger

from affine.model import Label, LanguageDetector

logger = getLogger(__name__)


def process_page(page):
    """ Runs langid's language detection on webpage text"""
    logger.info("Detecting language for page: %d"%page.id)
    lang_name = LanguageDetector.detect_language(page.title_and_text)
    lang_label = Label.by_name(lang_name)
    assert lang_label is not None, "Label %s does not exist"%lang_name

    det = LanguageDetector.query.one()
    LanguageDetector.delete_detector_results(page, [det.id])
    det.save_result(page.id, lang_label.id)

    return lang_label
