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
import json
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
    test_size=0.2, stratify=trainLabels,random_state=42)

(trainPaths, valPaths, trainLabels, valLabels) = split

datasets = [
    ('train', trainPaths, trainLabels, config.TRAIN_HDF5),
    ('val', valPaths, valLabels, config.VAL_HDF5)
]

(R, G, B) = ([], [], [])

for (dtype, paths, labels, outputPath) in datasets:

    print('[INFO] building {}...'.format(outputPath))

    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)
    
    for (path, label) in tqdm(zip(paths, labels)):
        try:
            image = cv2.imread(path)
        except:
            pass
        
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        if dtype == 'train':
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        writer.add([image], [label])
        
    writer.close()

print('[INFO] serializing means...')
D = {'R': np.mean(R), 'G':np.mean(G), 'B': np.mean(B)}
f = open(config.DATASET_MEAN, 'w')
f.write(json.dumps(D))
f.close()








