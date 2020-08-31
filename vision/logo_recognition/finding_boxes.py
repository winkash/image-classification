import os
import cv2

from affine.detection.model.features import SlidingWindowBoxExtractor


class BoxFinder(object):
    """ Finds logo boxes in an images
    Finds logo boxes in an image and return the list of the boxes
    with full path to each box_path

    Attributes:
        contrast_thresh : minimum contrast for boxes
        variance_thresh: minimum variance for boxes
        sliding_window_params:"explained in affine.detection.model.features.SlidingWindowBoxExtractor
    """

    def __init__(self, contrast_thresh=None, variance_thresh=None, **sliding_window_params):
        """Inits Box Finder class."""
        self.sliding_window_boxes = SlidingWindowBoxExtractor(**sliding_window_params)
        self.contrast_thresh = contrast_thresh
        self.variance_thresh = variance_thresh

    def get_boxes(self, out_dir, im_paths):
        """Returns boxes and their paths given a list of images

        Runs SlidingWindowBoxExtractor to get logo boxes for each image
        and returns dimensions of each box with the path to that box

        Args:
            out_dir : directory to save the returned box images
            im_paths: full path of input images

        Returns:
        list of the boxes with full path to each box_path
        """
        self.out_dir = out_dir
        extracted_boxes = self.sliding_window_boxes.extract(im_paths)
        embedded_boxes = []
        for idx, im_path in enumerate(im_paths):
            boxes = []
            img = cv2.imread(im_path)
            (height, width, _) = img.shape
            boxes += self.box_extractor(extracted_boxes, idx)
            embedded_boxes += self.write_boxes(boxes, img, im_path)
        return embedded_boxes

    def get_corner_boxes(self, height, width):
        """ Returns 4 corner boxes given height and width

        Args:
            width: box width
            height: box height

        Returns:
            list of 4 corner boxes
        """
        range = 3
        step_h = height / range
        step_w = width / range

        topleft = [step_h, step_w, 0, 0]
        topright = [step_h, step_w, 0, width - step_w]
        bottomleft = [step_h, step_w, height - step_h, 0]
        bottomright = [step_h, step_w, height - step_h, width - step_w]
        return [topleft, topright, bottomleft, bottomright]

    def box_extractor(self, extracted_boxes, idx):
        """Returns sliding window boxes for an image given the index of the image

        Args:
            idx: index if the image
            extracted_boxes: list of sliding window boxes for all the input images

        Returns:
            boxes: list of sliding window boxes of the image
        """
        boxes = []
        if idx in extracted_boxes.keys():
            boxes = extracted_boxes.get(idx)
        return boxes

    def quality_check(self, image):
        """ Checks the quality of the box

        Checks the quality of the box and returns True if the box is acceptable,
        otherwise returns False

        Args:
            image: the image in numpy.ndarray format
        Returns:
            True if the box is acceptable, False otherwise
        """
        diff = image.max() - image.min()
        if (self.variance_thresh is not None and image.var() <= self.variance_thresh)\
            or (self.contrast_thresh is not None and diff <= self.contrast_thresh):
            return False
        return True

    def _create_img_box(self, b, idx, img, im_path):
        """" creats a cropped image using the box_path

        crops the image using the box and write the box to a new image
        if it passes the quality Check
        Args:
            b: the box in [h, w, x, y] format
            idx: index of the box_path
            img: the image in numpy.ndarray format
            im_path: full path to image

        Returns:
            full path to the cropped image(box)
        """
        h, w, y, x = b
        img_cropped = img[y:y + h, x:x + w]
        box_path = None
        if self.quality_check(img_cropped):
            name_spl = os.path.split(im_path)[-1].split('.')
            box_name = os.path.basename(im_path).split('.')[0] + str(idx)
            box_path = os.path.join(self.out_dir, box_name + '.jpg')
            cv2.imwrite(box_path, img_cropped)
        return box_path

    def write_boxes(self, boxes, img,  im_path):
        """ write boxes to files and append their paths to the list of boxes

        creates a cropped image using the box dimensions, for each box in boxes list
        and append its path to the dimensions of the box and returns the embedded list

        Args:
            boxes: list of boxes of the image
            img: the image in numpy.ndarray format
            im_path: full path of the image

        Returns:
            list of boxes with their full path appended
        """
        results = []
        for idx, b in enumerate(boxes):
            box_path = self._create_img_box(b, idx, img, im_path)
            if box_path:
                b = b + [box_path]
                results.append(b)
        return results
