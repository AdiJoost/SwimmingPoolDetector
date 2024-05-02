import numpy as np
import cv2
import os

def main():
    print(getNumberOfYPictures());

def getNumberOfYPictures():
    for root, dirs, files in os.walk("./swissimage_annotator/static/data/swimmingpooldetector/y", topdown=False):
        return len(files)

def generateEqualDataSet(numberOfDataPoints: int):
    positiveGenerator = _getPathToPositivePicture()
    negativeGenerator = _getPathToNegativePicture()
    X = []
    y = []
    for _ in range(numberOfDataPoints + 1):
        X.append(_getPicture(positiveGenerator))
        y.append(1)
        X.append(_getPicture(negativeGenerator))
        y.append(0)
    return np.asarray(X), np.asarray(y)

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

if __name__ == "__main__":
    main()