import cv2
import numpy as np


__all__ = ["ImageDesc"]


class ImageDesc(object):

    def __init__(self, keypoints, descriptors):
        """ Constructor of ImageDesc object
            it takes a list of keypoints and a matrix of descriptors. It stores
            a matrix of the keypoint coordinates and its corresponding
            descriptors
        Args:
            keypoints: a list of keypoints (cv2.KeyPoint)
            descriptors: is a matrix, where row j correspond to a descriptor
            (computed on keypoint j)
        Assertion:
            AssertError if the number of keypoints and descriptors are not equal
        """

        self.cv_keypoints = keypoints
        self.key_size = len(keypoints)
        self.descriptors = descriptors
        if self.descriptors is not None:
            self.desc_rows = descriptors.shape[0]
            self.desc_cols = descriptors.shape[1]
        else:
            self.desc_rows = 0
            self.desc_cols = 0
        assert self.key_size == self.desc_rows, \
            "Number of keywpoint and descriptors is different"

        new_keypoints = []
        for k in self.cv_keypoints:
            new_keypoints.append([k.pt[0], k.pt[1]])
        self.keypoints = np.array(new_keypoints)

    def __eq__(self, other):
        if self.descriptors is None and self.cv_keypoints == []:
            return other.descriptors is None and other.cv_keypoints == []
        else:
            if self.keypoints.shape == other.keypoints.shape:
                return np.allclose(self.keypoints, other.keypoints) and \
                    np.allclose(self.descriptors, other.descriptors)
            else:
                return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def _get_kps_array(self):
        assert self.cv_keypoints is not None
        kps_array = np.zeros((self.key_size, 5))
        for row, kps in enumerate(self.cv_keypoints):
            for idx, attr in enumerate(['size', 'angle', 'response', 'octave', 'class_id']):
                kps_array[row][idx] = getattr(kps, attr)

        return kps_array

    def _get_cv_kps(self, kps_array):
        keypts = []
        for idx, (pt, keypt_attrs) in enumerate(zip(self.keypoints, kps_array)):
            x, y = pt
            size, angle, response, octave, class_id = keypt_attrs
            keypts.append(
                cv2.KeyPoint(x, y, size, angle, response, int(octave), int(class_id)))

        return keypts

    def __getstate__(self):
        cv_keypoints_array = self._get_kps_array()
        return self.keypoints, self.descriptors, cv_keypoints_array

    def __setstate__(self, state):
        self.keypoints, self.descriptors, cv_keypoints_array = state

        self.key_size = len(self.keypoints)
        if self.descriptors is not None:
            self.desc_rows = self.descriptors.shape[0]
            self.desc_cols = self.descriptors.shape[1]
        else:
            self.desc_rows = 0
            self.desc_cols = 0
        self.cv_keypoints = self._get_cv_kps(cv_keypoints_array)
