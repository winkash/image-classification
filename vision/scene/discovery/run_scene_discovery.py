import sys
import os
import argparse

from logging import getLogger

from affine.model import Label
from affine.detection.vision.scene.discovery.scene_discovery import SceneDiscovery

logger = getLogger("affine.discover_scenes")

def discover_scenes(config_file, folder, label_id):
    """ Returns a set of clusters that contain similar scenes for a given label
        Args:
            config_file: configuration file where all the descriptor and video parameters are defined
            folder: a directory for discovery's use
            label_id: id of the label to retrieve videos for (optional)
        Assertions:
            ValueError if the label id is not present in the DB
    """
    logger.info("starting scene discovery process")
    discover = SceneDiscovery(config_file, folder)
    pickle_folder = os.path.join(folder, 'pickles')
    image_folder = os.path.join(folder, 'images')
    feature_folder = os.path.join(folder, 'features')
    for fold in [pickle_folder, image_folder, feature_folder]:
        if not os.path.exists(fold):
            os.makedirs(fold)
    if label_id:
        if not Label.get(label_id):
            raise ValueError("Label id not found")
        discover.get_videos_from_inventory(label_id)
    else:
        discover.ingest_videos_youtube()
        discover.get_video_from_urls()
    discover.download_data(image_folder)
    discover.compute_descriptors(feature_folder)
    discover_file = os.path.join(pickle_folder, 'scenediscovery.pickle')
    SceneDiscovery.save_to_file(discover, discover_file)
    distances = ['intersection', 'chisqr']
    methods = ['single', 'ward', 'complete']
    for dist in distances:
        for meth in methods:
            discover.cluster_scenes(dist, meth)
            discover_file = os.path.join(pickle_folder, dist + '_' + meth + '_' + 'scenediscovery.pickle')
            SceneDiscovery.save_to_file(discover, discover_file)
            logger.info("saved clusters in %s " % discover_file)
    logger.info("Finished running scene discovery")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-inter_dir', action='store', dest='inter_dir', required=True,  help='set the folder for results')
    parser.add_argument('-config', action='store', dest='config', required=True, help='set the config path')
    parser.add_argument('-label_id', action='store', dest='label_id', required=False, type=int, help='label id')
    results = parser.parse_args()

    folder = os.path.normpath(results.inter_dir)
    discover_scenes(results.config, folder, results.label_id)
