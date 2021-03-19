from src import superpoint, utils
import argparse
import time
import os
from tqdm import tqdm
import cv2
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imgs_path', required=True,
                        help='input images path')
    parser.add_argument('--nms', type=int, default=2,
                        help='nms')
    parser.add_argument('--conf', type=float, default=0.015,
                        help='Block size for patterns.')
    parser.add_argument('--nn_tresh', type=float, default=1.0,
                        help='type of binarization.')
    parser.add_argument('--save_path', type=str, default="./",
                        help='path to save')
    return parser.parse_args()


def main():
    args = parse_args()
    nms_dist = args.nms
    conf_thresh = args.conf
    nn_thresh = args.nn_tresh
    images = sorted(os.listdir("data/"))
    keypoint_extractor = superpoint.SuperPointFrontend(weights_path="../SuperPointPretrainedNetwork/superpoint_v1.pth",
                                                       nms_dist=nms_dist, conf_thresh=conf_thresh, nn_thresh=nn_thresh)

    time_start = time.time()
    img_0 = cv2.imread(f"data/{images[0]}", 0)
    shape = img_0.shape
    img_0 = utils.prepare_img(img_0)
    pts_0, desc_0 = utils.extract_image_kp(img_0, keypoint_extractor)

    x_coef = shape[1] / 160
    y_coef = shape[0] / 120
    pts_0[0] *= x_coef
    pts_0[1] *= y_coef

    trans_phase = []
    trans = []
    for image in tqdm(images[1:]):
        img = cv2.imread(f"data/{image}", 0)
        shape = img.shape
        x_coef = shape[1] / 160
        y_coef = shape[0] / 120
        img = utils.prepare_img(img)
        (x_, y_), *_ = cv2.phaseCorrelate(img_0, img)
        trans_phase.append((round(x_ * x_coef, 1), round(y_ * y_coef, 1)))
        pts, desc = utils.extract_image_kp(img, keypoint_extractor)
        pts[0] *= x_coef
        pts[1] *= y_coef
        matches = superpoint.nn_match_two_way(desc_0, desc, nn_thresh)
        trans.append(utils.find_trans(pts_0, pts, matches))
    time_end = time.time()
    whole_time, time_one = utils.profiler(time_start, time_end, len(images))
    print("time for one image {: .3f} s, time for whole set {:.3f} s".format(time_one, whole_time))
    x_ss, y_ss, x_pc, y_pc = [], [], [], []
    for ss, pc in zip(trans, trans_phase):
        x_ss.append(ss[0])
        y_ss.append(ss[1])
        x_pc.append(pc[0])
        y_pc.append(pc[1])
    df = pd.DataFrame({"x superpoint": x_ss, "y superpoint": y_ss, "x phase correlation": x_pc, "y phase correlation": y_pc})
    print(df)
    df.to_csv(f"{args.save_path}/results.csv", index=False)

if __name__ == "__main__":
    main()