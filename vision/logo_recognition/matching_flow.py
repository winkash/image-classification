# from .training import CFG_SPEC
# from .model import LogoModel
from affine.detection.model.robust_matching import RobustMatcher
from affine.detection.model.mlflow import Flow, Step, \
    FutureFlowInput, FutureLambda


def logo_mathching_flow_factory(lm):
    """ This flow factory retuns a Flow which can perform Logo-Matching
    Args:
        lm : A LogoModel instance
    Returns:
        A Flow which can do logo-mathcing. The flow's kwargs for run_flow is
        `ip_data` ( The path to the logo-image )
    """
    rm = RobustMatcher(lm.min_points, lm.min_matches,
                       ransac_th=lm.ransac_th, accept_th=lm.accept_th,
                       ransac_algorithm=lm.ransac_algorithm,
                       ransac_max_iter=lm.ransac_max_iter,
                       ransac_prob=lm.ransac_prob, inlier_r=lm.inlier_r)
    f = Flow()
    feature_extract = Step("feat_ext", lm.bow, 'extract')
    knn = Step("knn", lm.knn, 'get_neighbors')
    get_near_logos = Step("acquire_near_logos",
                          lm.get_image_descs_and_target_label_ids)
    matching = Step("matching", rm, 'match')
    prediction = Step("predict", rm, 'predict')

    for step in [feature_extract, knn, get_near_logos, matching, prediction]:
        f.add_step(step)

    path_list = FutureLambda(FutureFlowInput(f, 'ip_data'), lambda x: [x])
    f.start_with(feature_extract, path_list)

    image_desc = FutureLambda(feature_extract.output, lambda x: x[0][0])
    hist = FutureLambda(feature_extract.output, lambda x: x[1][0])
    f.connect(feature_extract, knn, hist, lm.k_neighbors, dist=False)

    near_logo_ids = FutureLambda(knn.output, lambda x: x[0].tolist())
    f.connect(knn, get_near_logos, near_logo_ids)

    near_image_descs = FutureLambda(get_near_logos.output, lambda x: x[0])
    target_label_ids = FutureLambda(get_near_logos.output, lambda x: x[1])
    f.connect(get_near_logos, matching, image_desc, near_image_descs)

    f.connect(matching, prediction, matching.output, target_label_ids)

    f.output = prediction.output

    return f
