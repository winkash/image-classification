import os
import numpy as np

from affine.model import ResultAggregator
from affine.detection.vision.cnn_classifier.cnn_classifier_flow_factory\
    import CnnClassifierFlowFactory
from affine.video_processing import image_path_to_time, sample_images


def judge_video(image_dir, model_dir):
    """
    Evaluate if the video sampled in image_dir contains the targeted content.

    This method LOADS the classifier whose model files are given in model_dir
    everytime is called

    Args:
        image_dir: string, path to folder with video frames
        model_dir: string, path to folder with classifier model files
    Returns:
        image_results: dict with tuples (timestamp, boolean_result)
        video_result: boolean result
    Raises/Assertions:
        if the folder with images or folder with model files do not exist
    """
    assert (os.path.exists(image_dir)), \
        "Frames folder %s does not exist" % (image_dir)
    assert (os.path.exists(model_dir)), \
        "Model files folder %s does not exist" % (model_dir)

    ra = ResultAggregator()

    config_file = os.path.join(model_dir, 'Configfile.cfg')
    cnn_ff = CnnClassifierFlowFactory(config_file)
    cnn_ff.load_model(model_dir)

    max_num_frames = cnn_ff.max_frames_per_video
    f_predict = cnn_ff.get_process_video_flow()
    target_labels = cnn_ff.target_labels

    list_test_images = sample_images(image_dir, max_num_frames)
    num_frames = len(list_test_images)

    if num_frames:
        f_predict.run_flow(image_paths=list_test_images)
        predicted_labels = f_predict.output
        timestamps = [image_path_to_time(x) for x in list_test_images]

        results = {k: 0 for k in target_labels.keys()}
        for ts, clf_output in zip(timestamps, predicted_labels):
            clf_output = int(clf_output)
            if clf_output in target_labels.keys():
                tl = target_labels[clf_output]
                ra.add_image_result(ts, tl)
                results[clf_output] += 1

        for clf_output in target_labels.keys():
            tl = target_labels[clf_output]
            num_img_results = np.sum(results[clf_output])
            frames_current_tl = num_img_results / float(num_frames)
            video_result = (frames_current_tl >= cnn_ff.accept_th)\
                and (num_img_results >= cnn_ff.min_accept)
            if video_result:
                ra.add_video_result(tl)

    return ra.result_dict
