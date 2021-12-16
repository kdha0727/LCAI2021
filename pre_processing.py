import os
import numpy as np
import cv2
from PIL import ImageColor, Image
from xml.etree import ElementTree

import argparse
import sys

from constants import channel_map, img_height, img_width


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str,
                        default="raw_data", help='path to the root folder')
    parser.add_argument('--destination_folder', type=str,
                        default="processed_data", help='destination folder')
    return parser.parse_args(argv)


def process_case(src, dst, test):

    files = []
    if not os.path.exists(dst):
        os.makedirs(dst)

    for file in os.listdir(src):
        extension = file.split(".")[-1]
        if extension != 'xml':
            files.append(file)

    for file in files:
        # extension = file.split(".")[-1]
        filename = file.split(".")[0]
        print(file)
        if not os.path.exists(dst + "/image"):
            os.makedirs(dst + "/image")

        img_real = Image.open(os.path.join(src, file))
        img_real.load()
        width, height = img_real.size

        if not test:
            if not os.path.exists(dst + "/mask"):
                os.makedirs(dst + "/mask")
            if not os.path.exists(dst + "/mask_visualization"):
                os.makedirs(dst + "/mask_visualization")
            tree = ElementTree.parse(src + '/' + filename + ".xml")
            root = tree.getroot()
            root_size = root.findall("size")

            depth = int(root_size[0].findtext("depth"))
            mask = np.zeros((height, width, 7), dtype=np.uint8)
            mask_visualization = np.zeros((height, width, depth), dtype=np.uint8)
            shapes = root.findall("object")
            if not shapes:
                continue
            for shape in shapes:
                clr = shape.findtext("clr")
                if len(clr) == 9:
                    clr = clr[0] + clr[3:]
                points = shape.findall("points")
                data_x = points[0].findall("x")
                data_y = points[0].findall("y")

                r = []
                for point_x, point_y in zip(data_x, data_y):
                    r.append((int(float(point_x.text)),
                              int(float(point_y.text))))

                clr = ImageColor.getcolor(clr, "RGB")
                cur_mask = np.zeros((height, width), dtype=np.uint8)
                cur_mask = cv2.fillPoly(cur_mask, [np.asarray(r)], (1,), cv2.LINE_AA)
                try:
                    cur_channel = channel_map[clr]
                except KeyError as e:
                    print(e)
                    cur_channel = 0
                mask[:, :, cur_channel] = cur_mask
                for channel in range(7):
                    if channel != cur_channel:
                        mask[:, :, channel][mask[:, :, channel] == cur_mask] = 0
                cv2.fillPoly(mask_visualization, [np.asarray(r)], clr, cv2.LINE_AA)

            np.save(dst + "/mask/" + filename + ".npy", cv2.resize(mask, (img_width, img_height)))
            Image.fromarray(cv2.resize(mask_visualization, (img_width, img_height))).save(
                dst + "/mask_visualization/" + filename + ".png", format="png")

        small_img_real = cv2.resize(np.asarray(img_real), (img_width, img_height))  # noqa

        Image.fromarray(small_img_real).save(
            dst + "/image/" + filename + ".png", format="png")


def main(args):

    root_folder = os.path.expanduser(args.root_folder)
    destination_folder = os.path.expanduser(args.destination_folder)
    targets = ['/train_set/normal', '/train_set/benign_tumor', '/train_set/cancer', '/test_set_for_LCAI']
    tests = [False, False, False, True]
    for target, test in zip(targets, tests):
        process_case(root_folder + target, destination_folder + target, test)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
