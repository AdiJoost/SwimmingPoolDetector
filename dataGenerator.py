import numpy as np
import cv2
import os
import math

def main():
    print(getNumberOfYPictures());
    X,y = generateEqualDataSet(216, 0.3)
    print(X.shape)
    print(y.shape)

def getNumberOfYPictures():
    length = 0
    for root, dirs, files in os.walk("./swissimage_annotator/static/data/swimmingpooldetector/y", topdown=False):
        length += len(files)
    for root, dirs, files in os.walk("./poolPicturesValidated"):
        length += len(files)
    return length

def generateEqualDataSet(numberOfPositivDataPoints: int, trueFalseRatio=0.5):
    positiveGenerator = _getPathToPositivePicture()
    negativeGenerator = _getPathToNegativePicture()
    X = []
    y = []
    trueMultiplicator, falseMultiplicator = _getFalseMultiplicator(trueFalseRatio)
    iterations = math.floor(numberOfPositivDataPoints / trueMultiplicator)
    for _ in range(iterations):
        for _ in range(trueMultiplicator):
            X.append(_getPicture(positiveGenerator))
            y.append(1)
        for _ in range(falseMultiplicator):
            X.append(_getPicture(negativeGenerator))
            y.append(0)
    return np.asarray(X), np.asarray(y)

def _getPathToPositivePicture():
    for root, dirs, files in os.walk("./swissimage_annotator/static/data/swimmingpooldetector/y", topdown=False):
        for file in files:
            yield os.path.join(root, file)
    for root, dirs, files in os.walk("./poolPicturesValidated", topdown=False):
        for file in files:
            yield os.path.join(root, file)

def _getPathToNegativePicture():
    for root, dirs, files in os.walk("./swissimage_annotator/static/data/swimmingpooldetector/n", topdown=False):
        for file in files:
            yield os.path.join(root, file)

def _loadAndPreparePictureFromPath(path: os.path):
    image = cv2.imread(path)
    return np.asarray(image)

def _getFalseMultiplicator(trueFalseRatio):
    trueMultiplicator = math.floor(trueFalseRatio * 100)
    falseMultiplicator = 100
    gcd = np.gcd(trueMultiplicator, falseMultiplicator)
    return int(trueMultiplicator / gcd), int(falseMultiplicator / gcd)



def _getPicture(generator):
    picture = _loadAndPreparePictureFromPath(generator.__next__())
    picture = picture / 255
    return picture

if __name__ == "__main__":
    main()