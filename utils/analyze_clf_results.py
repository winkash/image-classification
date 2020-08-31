import argparse
from sqlalchemy.sql import func
from affine.model import session, MTurkImage, MTurkImageDetectorResult,\
    VideoOnPage, VideoHit, MTurkVideoDetectorResult, WebPageLabelResult, ClassifierTarget
from affine.model.detection import AbstractBetaDetector
from affine.model import AbstractClassifier, Label


def analyze_clf_results(clf, ct, mturkresults=MTurkImageDetectorResult):
    """
    Takes a classifier object (clf) and one of its classifier_target
    prints the following statistics:
    -Number of videos processed by beta clf
    -Total hits submitted
    -Total video results obtained
    -Total image results obtained
    -Image hits accuracy
    Args:
        clf: classifier object
        mturkresults: table of Mturk results to analyze
            (MTurkVideoDetectorResult or MTurkImageDetectorResult)
    Returns:
        a list of video_ids where the classifier had positive results
        (image or video results)
    """
    assert (ct in clf.clf_targets),\
        "Classifier target must be one within the targets in the given classifier"
    num_videos_processed = None

    if isinstance(clf, AbstractBetaDetector):
        num_videos_processed = len(clf.beta_video_detections)
        print 'Number of videos processed by beta clf =', num_videos_processed

    all_results_query = mturkresults.query.filter_by(clf_target=ct)
    ALL = all_results_query.count()
    print 'Total hits submitted = %d' % ALL

    list_vids = []
    if ct.video_qa_enabled:
        v_res = len(ct.video_results)
        print 'Total video results obtained', v_res
        print 'Video hits per day', ct.video_qa_count

    if mturkresults == MTurkImageDetectorResult and ct.image_qa_enabled:
        im_res = len(ct.image_results)
        print 'Total image results obtained', im_res
        print 'Image hits per day', ct.image_qa_count

        positive_results_query = all_results_query.join(MTurkImage).\
            filter(MTurkImage.result == True)
        TP = positive_results_query.count()
        negative_results_query = all_results_query.join(MTurkImage).\
            filter(MTurkImage.result == False)
        FP = negative_results_query.count()
        print 'Image hits accuracy %.2f' % (TP / float(TP + FP))
        list_vids = [
            res.mturk_image.video_id for res in positive_results_query]
        list_vids = list(set(list_vids))

    return list_vids


def find_labels_statistics(list_vids):
    """
    Gets a list of videos and prints the following statistics about it:
    -Most frequent label
    -Most frequent QAed label
    -Labels for video results sorted based on their frequency:
    -Labels for QAed video results sorted based on their frequency
    Args:
        List of video ids
    Returns:
        A tuple of webpage label ids and webpage label ids that have been QAed
    """
    pages_q = VideoOnPage.query.filter(VideoOnPage.video_id.in_(list_vids))
    page_ids = [p.page_id for p in pages_q]
    page_ids = list(set(page_ids))
    LABELID_IGNORE_LIST = [
        2853, 2854, 3077, 3078, 3079, 3080, 5639, 7840, 7972, 1054, 3076]

    def is_ignored(label_id):
        if label_id in LABELID_IGNORE_LIST or Label.get(label_id).label_type == 'flip':
            return True
        return False

    def clean_up_list(label_ids_list):
        return filter(lambda item: not is_ignored(item[0]), label_ids_list)

    label_ids_from_wpr = []
    if page_ids:
        label_ids_from_wpr = session.query(WebPageLabelResult.label_id,
                                     func.count(WebPageLabelResult.label_id)).\
            filter(WebPageLabelResult.page_id.in_(page_ids)).group_by(
                WebPageLabelResult.label_id).all()
    label_ids_from_wpr = clean_up_list(label_ids_from_wpr)

    label_ids_from_qawpr = []
    if page_ids:
        label_ids_from_qawpr = session.\
            query(VideoHit.label_id,
                  func.count(VideoHit.label_id)).\
            filter(VideoHit.page_id.in_(page_ids)).\
            filter(VideoHit.result == True).group_by(
                VideoHit.label_id).all()

    label_ids_from_qawpr = clean_up_list(label_ids_from_qawpr)

    label_ids_from_wpr.sort(key=lambda item: item[1])
    label_ids_from_qawpr.sort(key=lambda item: item[1])

    max_label = 'None'
    if label_ids_from_wpr:
        max_label = Label.get(label_ids_from_wpr[-1][0])
    max_label_qa = 'None'
    if label_ids_from_qawpr:
        max_label_qa = Label.get(label_ids_from_qawpr[-1][0])
    print 'Most frequent label = %s' % max_label
    print 'Most frequent QAed label = %s' % max_label_qa

    print 'Labels for video results sorted based on their frequency:'
    label_list = [Label.get(items[0]).name for items in label_ids_from_wpr]
    label_list.reverse()
    print '\n'.join(label_list)
    print 'labels for QAed video results sorted based on their frequency:'
    label_list_qa = [Label.get(items[0]).name for items in label_ids_from_qawpr]
    label_list_qa.reverse()
    print '\n'.join(label_list_qa)
    return (label_ids_from_wpr, label_ids_from_qawpr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--classifier_name', dest='classifier_name', required=True,
                        help='Name of the classifier')
    parser.add_argument('-t', '--clf_target_indexes', nargs='+', type=int,
                        dest='clf_target_indexes', help='Index of the classifier targets you want to see the result for. If none, all of the classifer targets will be analyzed')
    parser.add_argument('-m', '--mturk_results', type=str, dest='mturkresults',
                        help='MTurk Table to analyze. This can be either image or video')
    args = parser.parse_args()
    print args
    clf = AbstractClassifier.by_name(args.classifier_name)
    assert clf, "Classifier doesn't exist"

    classifier_target_indexes = args.clf_target_indexes
    assert len(
        clf.clf_targets), "The classifier doesn't have any classifier targets"

    if not classifier_target_indexes:
        classifier_target_list = clf.clf_targets
    else:
        classifier_target_list = [clf.clf_targets[index]
                                  for index in classifier_target_list]

    if args.mturkresults == 'image' or not args.mturkresults:
        mturkresults = MTurkImageDetectorResult
    elif args.mturkresults == 'video':
        mturkresults = MTurkVideoDetectorResult
    else:
        raise AssertionError("mturk_results has to be image or video")

    for ct in classifier_target_list:
        list_videos = analyze_clf_results(
            clf, ct=ct, mturkresults=mturkresults)
        if len(list_videos):
            find_labels_statistics(list_videos)

if __name__ == '__main__':
    main()
