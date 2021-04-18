from src import utils
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', required=True,
                        help='input images path')
    parser.add_argument('--output_dir', type=str, default="./",
                        help='path to save results graphs')
    return parser.parse_args()

def main():
    args = parse_args()
    methods = ["shi-thomasi", "sift", "orb"]
    img_list = utils.get_images(args.input_dir)
    print("descriptor based estimation of repeatability")
    results = {}
    # description based
    for method in methods:
        local_result = utils.compute_repeatability_descriptors(img_list, method)
        x_1 = []
        y_1 = []
        for i in range(11):
            y_1.append((local_result[0].sum(axis=1)>i).mean())
            x_1.append(i+2)
        y_2 = local_result[0].mean(axis=0)
        rep = local_result[0].mean()
        results[method] = {"time": local_result[1]*1000000, "repeatability":rep, "x":x_1, "y_1":y_1, "y_2":y_2}
        print(method, "{:.2f}".format(results[method]["repeatability"]))

    utils.plot_distribution(results, "portion of points that appear in Nth img mesuared by descriptors", "y_2",
                      "in_nth_images_matching", args.output_dir)
    utils.plot_distribution(results, "portion of points that appear in N or more imgs mesuared bydescriptors", "y_1",
                      "in_n_images_matching", args.output_dir)
    print()
    print("motion estimation based estimation of repeatability")
    results = {}
    for method in methods:
        local_result = utils.compute_repeatability_motion(img_list, method)
        x_1 = []
        y_1 = []
        for i in range(11):
            y_1.append((local_result[0].sum(axis=1) > i).mean())
            x_1.append(i + 2)
        y_2 = local_result[0].mean(axis=0)
        rep = local_result[0].mean()
        results[method] = {"time": local_result[1] * 1000000, "repeatability": rep, "x": x_1, "y_1": y_1, "y_2": y_2}
        print(method, "{:.2f}".format(results[method]["repeatability"]))
        utils.plot_distribution(results, "portion of points that appear in Nth img mesuared by motion estimation",
                          "y_2", "in_nth_images_motion_est", args.output_dir)
        utils.plot_distribution(results, "portion of points that appear in N or more imgs mesuared by motion estimation",
                          "y_1", "in_n_images_motion_est", args.output_dir)
    print()
    print("working time")
    for key in results:
        print(key, "{:.2f}".format(results[key]["time"]), "us")


if __name__ == "__main__":
    main()