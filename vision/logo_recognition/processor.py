import shutil
import cv2
import os
from tempfile import mkdtemp

from affine.detection.data_processor import DataProcessor
from affine.detection.model.mlflow import Step, Flow, ParallelFlow, FutureFlowInput, FutureLambda
from .finding_boxes import BoxFinder
from .matching_flow import logo_mathching_flow_factory
from .model import LogoModel


__all__ = ['LogoProcessor']


class LogoProcessor(DataProcessor):
    """Runs logo classification steps given a model directory

    Args:
        model_dir: model directory of the logos

    returns:
        List of [h, w, y, x, target_label_id]
    """

    def __init__(self, model_dir):
        """Initializes a LogoProcessor object given a model directory"""
        self.model_dir = model_dir
        self.lm = LogoModel(self.model_dir)

    @classmethod
    def load_model(cls, model_dir):
        return cls(model_dir)

    def predict(self, img_paths):
        """Runs end-to-end logo recognition.

        Args:
            img_paths: list of image paths

        Returns:
            List of [h, w, y, x, target_label_id]
        """
        pf = ParallelFlow(self._build_flow, max_workers=1)
        return pf.operate(img_paths)

    @staticmethod
    def resize_image(image_path, standard_width, out_dir):
        image_original = cv2.imread(image_path)
        height = image_original.shape[0]
        width = image_original.shape[1]
        ratio = standard_width/float(width)
        standard_height = int(height * ratio)
        image_resized = cv2.resize(image_original, (standard_width, standard_height))

        image_name = os.path.basename(image_path).split('.')[0]
        output_path = os.path.join(out_dir, image_name) + '.jpg'
        assert cv2.imwrite(output_path, image_resized)
        return ratio, output_path

    @staticmethod
    def _merger(boxes, target_label_ids, ratio=1):
        """Merges boxes with their corresponding label ids

        Args:
            boxes: list of boxes in [h, w, y, x] format
            label_ids: list of label ids

        Returns:
            list of [h, w, y, x, target_label_id]
        """
        labeled_boxes = []
        for b, l in zip(boxes, target_label_ids):
            if l != -1:
                box = b[0:-1]
                box = [int(b/ratio) for b in box]
                labeled_boxes.append(box + [l])
        return labeled_boxes

    def _build_flow(self):
        """Runs the logo detection flow

        Returns: logo detection flow which gets full path to an image as input
        """

        pf = ParallelFlow(logo_mathching_flow_factory, max_workers=3)
        f = Flow()
        setup = Step("setup", mkdtemp)
        resizer = Step("resizer", self.resize_image)
        bf = BoxFinder(contrast_thresh=self.lm.contrast_thresh, variance_thresh=self.lm.variance_thresh, patch_shapes=self.lm.patch_shapes,
                        scales=self.lm.scales, step_size=self.lm.step_size,
                        center_area_offset=self.lm.center_area_offset, corner_area_sz=self.lm.corner_area_sz, raise_on_size=self.lm.raise_on_size)
        box_finder = Step("box finder", bf, 'get_boxes')
        matcher = Step("matching flow", pf, 'operate')
        merger = Step("merger", self._merger)
        cleanup = Step("cleanup", shutil.rmtree)
        for s in [box_finder, matcher, merger, setup, cleanup]:
            f.add_step(s)
        # ip_data is full path to one single image
        img_path = FutureFlowInput(f, 'ip_data')
        img_list = FutureLambda(img_path, lambda x:[x])
        get_paths = lambda boxes: [b[-1] for b in boxes]
        paths = FutureLambda(box_finder.output, get_paths)
        number_boxes = FutureLambda(box_finder.output, lambda x: len(x))
        number_labels = FutureLambda(merger.output, lambda x: len(x))
        model_name =  os.path.basename(self.lm.model_dir)
        f.start_with(setup)
        if self.lm.resize:
            f.add_step(resizer)
            f.connect(setup, resizer, img_path, self.lm.standard_width, setup.output)
            resized_img_list = FutureLambda(resizer.output, lambda x:[x[1]])
            ratio = FutureLambda(resizer.output, lambda x:x[0])
            f.connect(resizer, box_finder, setup.output, resized_img_list)
            f.connect(matcher, merger, box_finder.output, matcher.output, ratio)
        else:
            f.connect(setup, box_finder, setup.output, img_list)
            f.connect(matcher, merger, box_finder.output, matcher.output)
        f.connect(box_finder, matcher, paths, self.lm)
        f.connect(merger, cleanup, setup.output)
        f.output = merger.output
        return f
