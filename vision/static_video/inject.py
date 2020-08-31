from affine.model import Label
from affine.detection.utils import AbstractInjector
from affine.model.detection import StaticVideoClassifier
from affine.detection.vision.static_video.static_video_classifier_ff import \
    StaticVideoClassifierFlowFactory
from affine.model import session


class StaticVideoClassifierInjector(AbstractInjector):

    def get_names(self):
        return StaticVideoClassifierFlowFactory.model_files

    def inject(self, clf_name, target_label_id_photo, target_label_id_sshow):
        '''
        Inject model to db.

        Params:
            clf_name: Name of classifier created.
            target_label_id_photo: Classifier's target label for photo videos
            target_label_id_sshow: target label for slideshow videos

        Returns:
            Classifier.
        '''
        target_labels_list = []
        for target_label_id in [target_label_id_photo, target_label_id_sshow]:
            l = Label.get(target_label_id)
            assert l, 'Label id %s does not correspond to any Label!' % \
                (target_label_id)
            target_labels_list += [l]

        assert not StaticVideoClassifier.by_name(clf_name), \
            'StaticVideoClassifier with name %s already exists!' % clf_name

        classifier = StaticVideoClassifier(name=clf_name)
        session.flush()

        classifier.add_targets(target_labels_list)
        self.tar_and_upload(classifier)
        return classifier
