import os
import tempfile

from affine.model import ResultAggregator
from affine.detection.vision.static_video import \
    StaticVideoClassifierFlowFactory as Svcff
from affine.detection.vision.video_motioncolor.video_motioncolor_classifier \
    import VideoInfoObject

from affine.model import Video


def judge_video(video_id, video_path, model_dir):
    """
    Evaluate if the video contains the targeted content,
    e.g., static photo, slideshow

    This method LOADS the classifier whose model files are given in model_dir
    everytime is called

    Args:
        video_id: long, video id
        video_path: string, full path name of the video file
        model_dir: string, path to folder with classifier model files
    Returns:
        A Boolean saying wether the detector fires (True) or not (False) for
        this video

    Raises/Assertions:
        Asserts if the video file passed is not there
    """
    assert (os.path.exists(video_path)), \
        "Video %s does not exist" % (video_path)

    ra = ResultAggregator()

    frames_folder = tempfile.mkdtemp()
    v = Video.get(video_id)
    if v:
        test_video_obj = VideoInfoObject(video_id=v.id, videopath=video_path,
                                         framespath=frames_folder,
                                         length=v.length)
        v_classif = Svcff.load_from_dir(model_dir)
        flow = v_classif.create_test_flow()

        video_result = flow.run_flow(video_obj=test_video_obj)
        if video_result in [Svcff.PHOTO, Svcff.SLIDESHOW]:
            ra.add_video_result(v_classif.target_labels[video_result])

    return ra.result_dict
