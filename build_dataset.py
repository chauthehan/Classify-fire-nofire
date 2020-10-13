from pyimagesearch.io import HDF5DatasetWriter
import numpy as np 
import argparse
from imutils import paths
import cv2
import os
import time
import imutils
import random
from tqdm import tqdm
import gc 
import config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--dataset", help="path to dataset")

args = vars(ap.parse_args())

trainPaths = list(paths.list_images(args["dataset"]))

random.shuffle(trainPaths)

trainLabels = [p.split(os.path.sep)[-2] for p in trainPaths]

trainLabels = LabelEncoder().fit_transform(trainLabels)

split = train_test_split(trainPaths, trainLabels, 
    test_size=)


