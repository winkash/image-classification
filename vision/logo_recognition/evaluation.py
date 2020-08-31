import argparse
import os
import shutil
import numpy as np
from collections import defaultdict
from configobj import ConfigObj
from tempfile import mkdtemp
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, \
    classification_report
from logging import getLogger

from affine.detection.vision.utils import scene_functions
from affine.detection.model.mlflow import Flow, Step, ParallelFlow

from affine.detection.vision.logo_recognition.logo import Logo
from affine.detection.vision.logo_recognition.training import CFG_SPEC, \
     train as train_logo_model
from affine.detection.vision.logo_recognition.matching_flow import \
    logo_mathching_flow_factory
from affine.detection.vision.logo_recognition.model import LogoModel


logger = getLogger(name=__name__)

__all__ = ['LogoEvaluator']


class LogoEvaluator(object):

    def __init__(self, config_file, n_folds=4):
        self.config_file = config_file
        self.n_folds = n_folds
        self.cfg = scene_functions.get_config(
            config_file, CFG_SPEC.split('\n'))
        self.base_dir = mkdtemp()
        logger.info("Base dir : %s", self.base_dir)

    def _list_iter(self, iterable, indices):
        return [item for idx, item in enumerate(iterable) if idx in indices]

    def split(self):
        splits = []
        logo_dir = self.cfg['logo_dir']
        assert os.path.isdir(logo_dir), "Logo dir does not exist !"
        logos = Logo.load_from_dir(logo_dir)

        paths = [l.path for l in logos]
        target_label_ids = [l.target_label_id for l in logos]

        skf = StratifiedKFold(target_label_ids, n_folds=self.n_folds)
        for idx, (train_idx, test_idx) in enumerate(skf):
            train_paths = self._list_iter(paths, train_idx)
            test_paths = self._list_iter(paths, test_idx)
            test_target_label_ids = self._list_iter(target_label_ids, test_idx)

            train_dir = os.path.join(self.base_dir, 'iteration%s' % idx)
            if not os.path.isdir(train_dir):
                os.makedirs(train_dir)

            # Since training applies target_label_ids on the fly based on names,
            # copying the logos with the same names will imply
            # that over all iterations the same logos will get the same target_label_ids
            # Not a good design, but will fix it soon
            for p in train_paths:
                _, fname = os.path.split(p)
                shutil.copy(p, os.path.join(train_dir, fname))

            # create a cfg file for this iteration
            train_cfg = self.cfg.copy()
            train_cfg['logo_dir'] = train_dir
            train_cfg['op_model_dir'] = os.path.join(
                self.base_dir, 'model%s' % idx)

            temp_cfg = ConfigObj()
            temp_cfg.update(train_cfg)
            temp_cfg.filename = os.path.join(
                self.base_dir, 'model%s.cfg' % idx)
            temp_cfg.write()

            splits.append(
                (temp_cfg.filename, train_cfg['op_model_dir'], test_paths, test_target_label_ids))

        return splits

    def evaluation_flow_factory(self, cfg_file, op_model_dir, test_paths):
        f = Flow()

        train = Step("training", train_logo_model)
        load = Step("model-load", lambda x: LogoModel(x))
        pf = ParallelFlow(logo_mathching_flow_factory, max_workers=3)
        test = Step("testing", pf, 'operate')

        for s in [train, load, test]:
            f.add_step(s)

        f.start_with(train, cfg_file)
        f.connect(train, load, op_model_dir)
        f.connect(load, test, test_paths, load.output)

        f.output = test.output
        return f

    def _train_and_test(self, cfg_file, op_dir, test_paths):
        f = self.evaluation_flow_factory(cfg_file, op_dir, test_paths)
        Y_pred = f.run_flow()
        return Y_pred

    def class_to_name(self):
        names_map = {-1 : "No Label"}
        logo_dir = self.cfg['logo_dir']
        logos = Logo.load_from_dir(logo_dir)
        for l in logos:
            if l.target_label_id not in names_map:
                names_map[l.target_label_id] = l.name

        return names_map

    def run_k_fold(self):
        metrics = defaultdict(list)
        splits = self.split()
        names_map = self.class_to_name()
        target_names = [names_map[v] for v in sorted(names_map.keys())]

        for idx, (cfg_file, op_dir, test_paths, Y_actual) in enumerate(splits):
            Y_pred = self._train_and_test(cfg_file, op_dir, test_paths)
            
            # copying testimages in trues and falses
            true_dir = os.path.join(self.base_dir, 'iteration%d' % idx, 'trues')
            false_dir = os.path.join(self.base_dir, 'iteration%d' % idx, 'falses')
            for dir in [true_dir, false_dir]:
                os.makedirs(dir)
            
            for y_t, y_p, path in zip(Y_actual, Y_pred, test_paths):
                bp = os.path.basename(path)
                if y_t == y_p:
                    shutil.copy(path, os.path.join(true_dir, bp))
                else:
                    shutil.copy(path, os.path.join(false_dir, bp))

            logger.info(classification_report(Y_actual, Y_pred, target_names=target_names))
            metrics['precision'].append(
                precision_score(Y_actual, Y_pred, average='weighted'))
            metrics['recall'].append(
                recall_score(Y_actual, Y_pred, average='weighted'))
            metrics['accuracy'].append(accuracy_score(Y_actual, Y_pred))

        for m in metrics:
            metrics[m] = np.array(metrics[m])
            print '%s %s\nMean = %0.2f' % (m, metrics[m], np.mean(metrics[m]))

        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file')
    parser.add_argument('n_folds', type=int)
    args = parser.parse_args()

    le = LogoEvaluator(args.cfg_file, args.n_folds)
    le.run_k_fold()
    logger.info("Finished Cross-validation !!")

if __name__ == '__main__':
    main()
