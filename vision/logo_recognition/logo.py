import cv2
import os
from logging import getLogger
from tempfile import mkdtemp
from uuid import uuid1

from affine.aws import s3client
from affine.model.labels import Label

logger = getLogger(name=__name__)

__all__ = ['Logo']


class Logo(object):

    _s3_bucket = 'set-classification-data'

    def __init__(self, s3_basename, target_label_id, name=None, path=None):
        self.s3_basename = s3_basename
        self.target_label_id = target_label_id
        self.name = name
        # local path
        self.path = path
        # feature attrs
        self.feature = None
        self.image_desc = None

    @classmethod
    def _construct_s3_urlpath(cls, s3_basename):
        return 'logo_recognition/uploaded_logos/%s' % s3_basename

    @property
    def _s3_urlpath(self):
        return self._construct_s3_urlpath(self.s3_basename)

    @property
    def s3_path(self):
        if self.s3_basename:
            return 'https://s3.amazonaws.com/%s/%s' % (self._s3_bucket, self._s3_urlpath)

    def __repr__(self):
        return u"<Logo name=%s, target_label_id=%s, s3_path=%s>" % \
            (self.name, self.target_label_id, self.s3_path)

    def download_from_s3(self, path):
        assert self.s3_basename
        s3client.download_from_s3(self._s3_bucket, self._s3_urlpath, path)

    @classmethod
    def upload_to_s3(cls, path, s3_name):
        ext = '.jpg'
        assert path.endswith(ext)
        s3_basename = '%s_%s%s' % (s3_name, uuid1(), ext)
        urlpath = cls._construct_s3_urlpath(s3_basename)
        s3client.upload_to_s3(cls._s3_bucket, urlpath, path, True)
        return s3_basename

    @classmethod
    def resize(cls, img_path, out_dir, max_size):
        img = cv2.imread(img_path)
        ht, wt = img.shape[0], img.shape[1]
        if ht > wt:
            ratio = float(max_size) / ht
        else:
            ratio = float(max_size) / wt
        new_ht = ht * ratio
        new_wt = wt * ratio

        image_resized = cv2.resize(img, (int(new_wt), int(new_ht)))

        name = os.path.basename(img_path).split('.')[0]
        output_path = os.path.join(out_dir, name) + '.jpg'
        cv2.imwrite(output_path, image_resized)
        return output_path

    @classmethod
    def load_from_dir(cls, logo_dir, separator='__', create_logo_labels=True,
                      max_logo_size=None):
        """Load logos from a direcotry containing all the logo images.
        The returned logo objects lack s3_paths.
        Real target_label_ids are assgined if create_logo_labels is True
        These must be set if the logo is to be injected in production.

        Args:
            logo_dir : The directory containing all the images
            separator : The separator used in the names of the logo files
            create_logo_labels : Creates new labels if label with the name
                does not exists.
            max_logo_size : resize logos s.t. max dimension of the logo
                is equal to max_logo_size.

        Returns:
            A list of logo objects

        Warnings:
            The method ignores all files in the logo_dir that do not have
            a .jpg extension
        """
        logos = []
        resized_logo_dir = mkdtemp()
        for img in sorted(os.listdir(logo_dir)):
            if not img.endswith('.jpg'):
                logger.warning("%s does not end with .jpg, Ignoring", img)
                continue

            names = img.split(separator)
            assert len(names) == 2, \
                "More than i separator found in logo-name, %s" % img
            label_name = names[0] + '-logo'
            logo_label = Label.by_name(label_name)
            if not logo_label:
                if create_logo_labels:
                    logo_label = Label.get_or_create(label_name)
                else:
                    raise ValueError("Label does not exists : %s" % label_name)

            local_path = os.path.join(logo_dir, img)
            if max_logo_size is not None:
                local_path = cls.resize(local_path,
                                        resized_logo_dir, max_logo_size)

            # logo is not in s3
            s3_basename = None

            logo = cls(s3_basename, logo_label.id, name=names[0],
                       path=local_path)
            logos.append(logo)

        return logos
