from datetime import datetime

from affine.model._sqla_imports import *
from affine.model.detection import AbstractBetaDetector, ResultAggregator,\
    LogoClassifierMixin

__all__ = ['SpatialSceneBetaDetector', 'CnnBetaClassifier',
           'TextDetectBetaClassifier', 'TextRecognizeBetaClassifier',
           'LogoBetaClassifier']


class SpatialSceneBetaDetector(AbstractBetaDetector):
    __tablename__ = "spatial_scene_beta_detectors"
    __mapper_args__ = dict(polymorphic_identity='spatial_scene_beta_detector')

    id = Column('id', Integer, ForeignKey('abstract_beta_detectors.id'),
                 primary_key=True)
    video_threshold = Column(Integer, nullable=False, default=2)
    image_threshold = Column(Float)

    def judge_video(self, video_path=None, imagedir=None):
        from affine.detection.vision.spatial_scene import detection as ssd
        res = ssd.judge_video(imagedir, self.local_dir(), self.video_threshold,
                               self.image_threshold)
        return self._process_results(*res)


class CnnBetaClassifier(AbstractBetaDetector):

    """ Cnn feature based BETA classifier """
    __mapper_args__ = dict(polymorphic_identity='cnn_beta_classifier')

    def judge_video(self, video_path=None, imagedir=None):
        from affine.detection.vision.cnn_classifier \
            import detection as cnn_detection
        res = cnn_detection.judge_video(
            image_dir=imagedir, model_dir=self.local_dir())
        return res


class LogoBetaClassifier(LogoClassifierMixin, AbstractBetaDetector):
    __mapper_args__ = dict(polymorphic_identity="logo_beta_classifier")


class TextDetectBetaClassifier(AbstractBetaDetector):

    """ In-Image Text detector in the wild BETA """
    __tablename__ = "text_detect_beta_classifier"
    __mapper_args__ = dict(polymorphic_identity='text_detect_beta_classifier')

    id = Column('id', Integer, ForeignKey('abstract_beta_detectors.id'),
                 primary_key=True)
    pred_thresh = Column(Float, nullable=False, default=0.7)
    added = Column(DateTime, default=datetime.utcnow)

    def _process_results(self, bounding_rects):
        ra = ResultAggregator()
        for time in bounding_rects.keys():
            for h, w, x, y in bounding_rects[time]:
                ra.add_new_box(x, y, w, h, time,
                               'Text', label_id=self.clf_target.target_label_id)

        if bounding_rects:
            ra.add_video_result(self.clf_target.target_label_id)

        return ra.result_dict

    def judge_video(self, video_path=None, imagedir=None):
        from affine.detection.vision.vision_text_detect \
            import detection as vision_text_detection

        bounding_rects = vision_text_detection.judge_video(
            imagedir=imagedir, model_dir=self.local_dir(),
            word_det_th=self.pred_thresh)

        return self._process_results(bounding_rects)


class TextRecognizeBetaClassifier(AbstractBetaDetector):

    """In-Image Text recognizer in the wild BETA """
    __tablename__ = "text_recognize_beta_classifier"
    __mapper_args__ = dict(
        polymorphic_identity='text_recognize_beta_classifier')

    id = Column('id', Integer, ForeignKey('abstract_beta_detectors.id'),
                 primary_key=True)
    pred_thresh = Column(Float, nullable=False, default=0.2)
    added = Column(DateTime, default=datetime.utcnow)

    def process_video(self, video_id, video_path=None, imagedir=None):
        """Takes in a video id to find text boxes to recognize."""
        words = self.judge_video(video_id, video_path, imagedir)
        self.save_results(words)

    def judge_video(self, video_id, video_path=None, imagedir=None):
        from affine.detection.vision.vision_text_recognize \
            import detection as vision_text_recognize
        return vision_text_recognize.judge_video(
            video_id=video_id, imagedir=imagedir, model_dir=self.local_dir(),
            rec_th=self.pred_thresh)

    def save_results(self, words):
        from affine.detection.vision.vision_text_recognize \
            import detection as vision_text_recognize
        vision_text_recognize.save_results(words)
