import sys
import os
import argparse

from logging import getLogger

from affine.detection.vision.scene.recognition.scene_classifier import SceneClassifier

__all__= ['scene_recognition_discovery']

logger = getLogger(__name__)

TPR = 0.8
FPR = 0.1

def scene_recognition_discovery(config_filename, inter_dir, posfile, negfile=None):
    """ creates a scenedetector from begining to end 
        Args:
            config_filename: string, path to scene config file
            inter_dir: string, path to folder where all the files will be saved
            posfile: string, path to file specifying positive data
            negfile: string, path to file specifying negative data
    """

    scene = SceneClassifier(config_filename, inter_dir)
    scene.set_params()
    logger.info("obtaining negative dataset")
    new_negfile = scene.write_negative_examples(negfile)
    logger.info("creating datasets")
    scene.create_dataset_partitions(posfile, new_negfile)
    logger.info("training")
    scene.train_classifier()
    logger.info("testing")
    results = scene.test_classifier()
    logger.info("Evaluating")
    scene.evaluate_classifier(results)
    logger.info("Injecting classifier")
    if scene.tpr >= TPR and scene.fpr <= FPR:
        scene.inject_classifier()
    else:
        logger.info('Retrain classifier. TPR = %f , FPR = %f' % (scene.tpr, scene.fpr))
    logger.info("Finished running scene recognition discovery")

if __name__ == '__main__':
    # parsing the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-inter_dir', action='store', dest='inter_dir', required=True, help='set the inter_dir trained detector')
    parser.add_argument('-config', action='store', dest='config_filename', required=True, help='set the config filename')
    parser.add_argument('-posfile', action='store', dest='posfile', required=True, help='path to file specifying positive data')
    parser.add_argument('-negfile', action='store', dest='negfile', required=False, help='path to file specifying negative data')
    results = parser.parse_args()

    for file_path in [results.inter_dir, results.config_filename, results.posfile]:
        if not os.path.exists(file_path):
            print '%s not found!' % file_path
            sys.exit(1)

    if results.negfile and not os.path.exists(results.negfile):
        print '%s not found!' % results.negfile
        sys.exit(1)

    scene_recognition_discovery(results.config_filename, results.inter_dir, results.posfile, results.negfile)
