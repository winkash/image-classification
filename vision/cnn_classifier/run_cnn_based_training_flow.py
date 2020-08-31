import os
import argparse
import numpy as np

from logging import getLogger

from affine.model import Label
from affine.detection.model.mlflow.flow \
    import Step, Flow, FutureLambda, FutureFlowInput
from affine.detection.vision.utils.scene_functions import get_config
from affine.detection.vision.cnn_classifier.cnn_classifier_flow_factory \
    import CnnClassifierFlowFactory
from affine.detection.vision.cnn_classifier.inject_detector_to_db \
    import CnnClassifierInjector, CnnBetaClassifierInjector

logger = getLogger(__name__)

TRAINCNNCLF_CFG_SPEC = """
    [train_params]
        training_folders = string_list(min=2, default=list('negative', 'positive'))
        target_label_ids = int_list(default=list())
    """


def acquire_data(training_path, list_labeled_folders=[]):
    """
    Get training images and labels from the folder provided with training data.

    This folder should contain subfolders, one per label, with training images.
    For training, this function assigns the same int label to all images
    within the same folder.

    NOTE that this automatically assigned int is just the training label.
    Later, we need to stablish the relationship between each of them
    and "real" Label ids from our inventory if it applies.

    Args:
        training_path: string,
            path to folder with training data arranged in subfolders.
        list_labeled_folders: list of strings,
            each one corresponds to the subfolder name of each label,
            e.g., ['label1', 'label2', 'label3']
            if empty, the default configuration expected is
            ['positive', 'negative']:
                folder 'positive': with different files and/or folders
                    (containing positive training data)
                folder 'negative': similar but with negative examples
    Returns:
        numpy array with pairs [image_path, label] in each row

    """

    if list_labeled_folders:
        training_label_list = []
        label_ids_dict = {}
        for n, l in enumerate(list_labeled_folders):
            training_label_list += [l]
            label_ids_dict.update({l: n})
    else:
        l0 = 'negative'
        l1 = 'positive'
        training_label_list = [l0, l1]
        label_ids_dict = {l1: 1, l0: 0}

    all_labels = []
    all_file_paths = []
    for l in training_label_list:
        datapath = os.path.join(training_path, l)
        assert os.path.exists(datapath), \
            "Training path should contain a folder named %s" % l

        n_currentfiles = 0
        for root, dirs, files in os.walk(datapath):
            img_files = [os.path.join(root, f) for f in files
                         if f.endswith('.jpg')]
            all_file_paths += img_files
            n_currentfiles += len(img_files)

        all_labels += [label_ids_dict[l]] * n_currentfiles

    all_labels = np.array(all_labels, ndmin=2).transpose()
    all_file_paths = np.array(all_file_paths, ndmin=2).transpose()
    data_files = np.append(all_file_paths, all_labels, axis=1)

    return data_files


def inject(model_dir, detector_name, list_label_ids, true_vid_list, beta,
           pickle_file):
    """
    Inject a classifier into the DB

    Args:
        model_dir: path to folder with classifier files required by injector
        detector_name: string, name of the classifier (has to be unique)
        list_label_ids: list of label ids that are targets of this classifier.
            A ClassifierTarget entry will be created for each one.
        true_vid_list: list of video ids used for training this classifier
        beta: bool, True if the classifier should be a beta classifier

    Returns:
        clf: classifier object injected
    """
    beta_msg = ''
    if beta:
        cnn_injector = CnnBetaClassifierInjector(model_dir,
                                                 optional_files=[pickle_file])
        beta_msg = 'BETA '
    else:
        cnn_injector = CnnClassifierInjector(model_dir,
                                             optional_files=[pickle_file])
    clf = cnn_injector.inject_detector(detector_name, list_label_ids,
                                       true_vid_list)
    logger.info('%sClassifier %s injected to the DB' % (beta_msg, clf.name))
    return clf


def evaluate_and_inject(metrics, model_dir, detector_name, list_label_ids,
                        true_vid_list, crossval_accept, beta, pickle_file):
    """ Helper method (Step) to evaluate the cross validation metrics
    and inject the classifier if metric results are above accept thresholds """

    clf = None
    if np.mean(metrics['accuracy']) > crossval_accept:
        clf = inject(model_dir, detector_name, list_label_ids, true_vid_list,
                     beta, pickle_file)
    else:
        logger.info('Classifier not injected to the DB due to low accuracy')
    return clf


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-training_data_path",
                        help="path to find the nudity dataset",
                        default=None)
    parser.add_argument("-configfile",
                        help="path to configfile, e.g., ./data/Configfile.cfg")
    parser.add_argument(
        "-classifier_name",
        help="unique classifier name in the DB if training is successful")
    parser.add_argument(
        '-training_vids_file', default=None,
        help="path to numpy file with list of training video ids")
    parser.add_argument(
        "-model_dir", default="/tmp/model_dir",
        help="full path folder name to save or read trained model")
    parser.add_argument('--beta', dest='beta', action='store_true',
                        help="inject as beta classifier")
    parser.set_defaults(beta=False)

    return parser.parse_args(args)


def load_training_config(configfile_name):
    """
    Load the part of the config file relative to training of this classifier
    """
    training_folders = []
    cfg_obj = get_config(
        configfile_name, spec=TRAINCNNCLF_CFG_SPEC.split('\n'))
    if cfg_obj:
        training_folders = cfg_obj['train_params']['training_folders']
        target_labels_info = cfg_obj['train_params']['target_label_ids']
        label_ids = target_labels_info[1::2]
        for l_id in label_ids:
            assert Label.get(l_id), \
                " Target label id %d does not exist in the DB" % (l_id)

    return training_folders


def main(args=None):
    """
    Run training flow to obtain (train and/or inject) a cnn-feature based classifier 

    NOTE:
    - If we want to only inject:
    1) training_data_path should not be provided (or being None) given given,
    2) this script will expect the model files required to be injected in the
    "model_dir" path given.

    - If we want to train and inject:
    1)the given training_data_path should include one folder for each label,
    e.g., a folder called 'negative' and another called 'positive',
    each of them with the images/folders containing corresponding examples
    2) the model will be stored in the given "model_dir" path.

    For more details on the options, run this with -h

    """
    args = parse_args(args=args)
    root_path = args.training_data_path

    training_vids = []
    if args.training_vids_file and os.path.exists(args.training_vids_file):
        training_vids = np.load(args.training_vids_file)
    training_folders = load_training_config(configfile_name=args.configfile)
    cnn_ff = CnnClassifierFlowFactory(configfile_name=args.configfile)
    pickle_file = cnn_ff.CLASSIFIER_FILE[cnn_ff.clf_type]
    if not args.training_data_path:
        inject(args.model_dir, args.classifier_name,
               cnn_ff.target_labels.values(), training_vids, args.beta,
               pickle_file=pickle_file)
        return None
    f_train = cnn_ff.get_train_flow()
    data_grabber = Step('data', acquire_data, None)
    training_flow = Step('training', f_train, 'run_flow')
    inject_classif = Step('inject', evaluate_and_inject, None)

    f = Flow('Cnn feature based classifier training')
    for step in [data_grabber, training_flow, inject_classif]:
        f.add_step(step)

    f.start_with(data_grabber, FutureFlowInput(f, 'training_path'),
                 FutureFlowInput(f, 'list_labeled_folders'))
    image_paths = FutureLambda(
        data_grabber.output, lambda x: [row[0] for row in x])
    image_labels = FutureLambda(
        data_grabber.output,
        lambda x: np.asarray([row[1] for row in x], dtype=int))
    f.connect(data_grabber, training_flow,
              image_paths=image_paths, image_labels=image_labels,
              output_model_dir=args.model_dir)
    f.connect(training_flow, inject_classif, metrics=training_flow.output,
              model_dir=args.model_dir,
              detector_name=args.classifier_name,
              list_label_ids=cnn_ff.target_labels.values(),
              true_vid_list=training_vids,
              crossval_accept=cnn_ff.crossval_accept,
              beta=args.beta,
              pickle_file=pickle_file)
    f.output = training_flow.output

    f.run_flow(training_path=root_path,
               list_labeled_folders=training_folders)

    crossval_metrics = f.output

    return crossval_metrics

if __name__ == '__main__':
    main()
