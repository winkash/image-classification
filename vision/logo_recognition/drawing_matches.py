import argparse
import os
import pickle

from affine.detection.model.mlflow import FlowTester
from affine.detection.vision.logo_recognition.model import LogoModel
from affine.detection.vision.logo_recognition.finding_boxes import BoxFinder
from affine.detection.vision.logo_recognition.matching_flow import logo_mathching_flow_factory
from affine.detection.model.robust_matching import RobustMatcher


def draw_matches(img_path, model_dir, out_dir, logo_dir):

	lm = LogoModel(model_dir)
	bf = BoxFinder(contrast_thresh=lm.contrast_thresh, variance_thresh=lm.variance_thresh,
	               patch_shapes=lm.patch_shapes, scales=lm.scales, step_size=lm.step_size,
	               center_area_offset=lm.center_area_offset, corner_area_sz=lm.corner_area_sz,
	               raise_on_size=lm.raise_on_size)
	boxes_out_dir = os.path.join(out_dir, 'boxes')
	if not os.path.isdir(boxes_out_dir):
		os.mkdir(boxes_out_dir)
	bf_results = bf.get_boxes(boxes_out_dir, [img_path])
	box_paths = [res[-1] for res in bf_results]
	ft = FlowTester(logo_mathching_flow_factory, lm)
	logos_path = os.path.join(model_dir, 'logos')
	knn_path = os.path.join(model_dir, 'knn')
	with open(logos_path, 'r') as f:
		logos = pickle.load(f)
	with open(knn_path, 'r') as f:
		knn = pickle.load(f)
	for box_path in box_paths:
		feat_ext_output = lm.bow.extract([box_path])
		image_desc = feat_ext_output[0][0]
		hist = feat_ext_output[1][0]
		knn_output = knn.get_neighbors(hist, lm.k_neighbors, dist=False)
		near_logo_ids = knn_output[0].tolist()
		near_logos_imgdesc = lm.get_image_descs_and_target_label_ids(near_logo_ids)[0]
		candidate_old_paths = [logos[number].path for number in near_logo_ids]
		candidate_names = [os.path.basename(cnd) for cnd in candidate_old_paths]
		candidate_paths = [os.path.join(logo_dir, cnd_name) for cnd_name in candidate_names]
		rm = RobustMatcher(min_points=lm.min_points, min_matches=lm.min_matches,
	                   ransac_th=lm.ransac_th, accept_th=lm.accept_th, debug=True, ransac_algorithm = lm.ransac_algorithm)
		scores = rm.match(image_desc, near_logos_imgdesc)
		rm.draw_matches(box_path, candidate_paths, scores, out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, dest='img_path', required=True,
                        help='Query image path')
    parser.add_argument('-m', '--model-dir', type=str, dest='model_dir', required=True,
                        help='Directory that contains the model files')
    parser.add_argument('-o', '--out_dir', type=str, dest='out_dir', required=True,
						help='Directory to save drawings')
    parser.add_argument('-l', '--logo_dir', type=str, dest='logo_dir', required=True,
                        help='Directory to save logos')
    args = parser.parse_args()
    args_list = [args.img_path, args.model_dir, args.out_dir, args.logo_dir]
    for arg in args_list:
    	assert os.path.exists(arg), arg+' does not exist!'

    draw_matches(*args_list)

if __name__ == '__main__':
    main()
