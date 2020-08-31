import os
import tempfile

from affine.detection.vision.video_motioncolor.video_motioncolor_classifier \
        import VideoMotionColorClassifier, VideoInfoObject

from affine.model import Video


def judge_video(video_id, video_path, model_dir, conf_th, accept_th):
    """
    Evaluate if the video contains the targeted content,
    e.g., a videogame play, or not.

    This method LOADS the classifier whose model files are given in model_dir
    everytime is called

    Args:
        video_id: long, video id
        video_path: string, full path name of the video file
        model_dir: string, path to folder with classifier model files
        conf_th, accept_th: acceptance thresholds from the config of
        VideoMotionColorClassifier
    Returns:
        A Boolean saying wether the detector fires (True) or not (False) for
        this video

    Raises/Assertions:
        Asserts if the video file passed is not there
    """
    assert (os.path.exists(video_path)), \
        "Video %s does not exist" % (video_path)

    VIDEO_TARGET_LABEL = 0
    video_result = False
    predictions = []

    frames_folder = tempfile.mkdtemp()

    v = Video.get(video_id)

    if v:
        test_videoObj = VideoInfoObject(
            video_id=v.id, videopath=video_path, framespath=frames_folder,
            length=v.length)

        config_file = os.path.join(model_dir, 'Configfile.cfg')
        v_classif = VideoMotionColorClassifier(config_file)
        v_classif.load_model(model_dir)

        if conf_th <= 1:
            v_classif.confidence_th = conf_th
        if accept_th <= 1:
            v_classif.accept_th = accept_th

        predictions = v_classif.classify_videos([test_videoObj])

        video_result = (predictions[0] == VIDEO_TARGET_LABEL)

    return {}, video_result
