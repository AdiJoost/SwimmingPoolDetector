import numpy as np
import cv2
import os

def generateEqualDataSet(numberOfDataPoints: int):
    positiveGenerator = _getPathToPositivePicture()
    negativeGenerator = _getPathToNegativePicture()
    dataSet = []
    for _ in range(numberOfDataPoints + 1):
        dataSet.append((_getPicture(positiveGenerator), 1))
        dataSet.append((_getPicture(negativeGenerator), 0))
    np.random.shuffle(dataSet)
    return dataSet

def _getPathToPositivePicture():
    for root, dirs, files in os.walk("./swissimage_annotator/static/data/swimmingpooldetector/y", topdown=False):
        for file in files:
            yield os.path.join(root, file)

def _getPathToNegativePicture():
    for root, dirs, files in os.walk("./swissimage_annotator/static/data/swimmingpooldetector/n", topdown=False):
        for file in files:
            yield os.path.join(root, file)

def _loadAndPreparePictureFromPath(path: os.path):
    image = cv2.imread(path)
    return np.asarray(image)

def _getPicture(generator):
    picture = _loadAndPreparePictureFromPath(generator.__next__())
    picture = picture / 255
    return picture