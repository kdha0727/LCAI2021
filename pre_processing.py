import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2

import argparse
import sys
from PIL import ImageColor, Image


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str,
                        default="raw_data", help='path to the root folder')
    parser.add_argument('--destination_folder', type=str,
                        default="processed_data", help='destination folder')
    parser.add_argument('--test', action='store_true',
                        help='whether trnsform is test')

    parser.add_argument('--img_height', type=int, default=256,
                        help='height of the result images')
    parser.add_argument('--img_width', type=int, default=384,
                        help='width of the result images')
    return parser.parse_args(argv)


# root_folder="raw_data"
# destination_folder="processed_data"

def main(args):
    root_folder = os.path.expanduser(args.root_folder)
    destination_folder = os.path.expanduser(args.destination_folder)
    img_height = args.img_height
    img_width = args.img_width
    test_mode = args.test

    files = []
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file in os.listdir(root_folder):
        extension = file.split(".")[-1]
        if extension != 'xml':
            files.append(file)

    for file in files:
        extension = file.split(".")[-1]
        filename = file.split(".")[0]
        print(file)
        if not os.path.exists(destination_folder + "/image"):
            os.makedirs(destination_folder + "/image")

        img_real = Image.open(root_folder+"/"+file)
        img_real.load()
        width, height = img_real.size

        if not test_mode:
            if not os.path.exists(destination_folder + "/mask"):
                os.makedirs(destination_folder + "/mask")
            tree = ET.parse(root_folder+'/'+filename+".xml")
            root = tree.getroot()
            root_size = root.findall("size")

            depth = int(root_size[0].findtext("depth"))
            mask = np.zeros((height, width, depth), dtype=np.uint8)
            shapes = root_size = root.findall("object")
            if shapes == []:
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

                cv2.fillPoly(mask, [np.asarray(r)], clr, cv2.LINE_AA)

            small_mask = cv2.resize(mask, (img_width, img_height))
            small_mask = Image.fromarray(small_mask).save(
                destination_folder+"/mask/"+filename+".png", format="png")

        small_img_real = cv2.resize(np.asarray(
            img_real), (img_width, img_height))

        small_img_real = Image.fromarray(small_img_real).save(
            destination_folder+"/image/"+filename+".png", format="png")


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))