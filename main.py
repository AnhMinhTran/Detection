import argparse
import time

import cv2 as cv
import imutils
import numpy as np
import timm
import torch
import torchvision
from torchvision.transforms import transforms

from utils import img_pyramid
from utils import slidind_win

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-s", "--size", type=str, default="(200, 150)",
                help="ROI size (in pixels)")
ap.add_argument("-c", "--min-conf", type=float, default=0.9,
                help="minimum probability to filter weak detections")
ap.add_argument("-v", "--visualize", type=int, default=-1,
                help="whether or not to show extra visualizations for debugging")
args = vars(ap.parse_args())

# initialize variables used for the object detection procedure
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (224, 224)

print("[INFO] loading network...")
model = timm.create_model('efficientnet_v2s', num_classes=7)
checkpoint = torch.load("models/model_best.pth-d043d179.pth")
model.load_state_dict(checkpoint)
model.eval()
print('something')
orig = cv.imread(args["image"])
orig = imutils.resize(orig, width=WIDTH)
(H, W) = orig.shape[:2]

pyramid = img_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)

rois = []
locs = []

start = time.time()

for image in pyramid:
    scale = W / float(image.shape[1])

    for (x, y, roiOrig) in slidind_win(image, WIN_STEP, ROI_SIZE):
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)

        roi = cv.resize(roiOrig, INPUT_SIZE)

        transformation = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        roi = np.asarray(roi)
        roi = torchvision.transforms.ToTensor()(roi)

        rois.append(roi)
        locs.append((x, y, x + w, y + h))

        if args['visualize'] > 0:
            clone = orig.copy()
            cv.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv.imshow("Visualization", clone)
            cv.imshow("ROI", roiOrig)
            cv.waitKey(0)
end = time.time()

print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(
    end - start))
print(type(rois))
# rois = np.array(rois, dtype="float32")
print("[INFO] classifying ROIS")
start = time.time()
preds = model.eval(rois)
end = time.time()
print("[INFO] classifying ROIs took {:.5f} seconds".format(
    end - start))
