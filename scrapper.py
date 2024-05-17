import os
import logging
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

def main():
    scrapModel("vgg16.h5")


def scrapModel(modelName):
    _initScrapper(modelName)
    model = _getModel(modelName)
    _runThroughPictures(model, modelName)

def _initScrapper(modelName):
    _configureLogger()
    _validateFolders(modelName)

def _configureLogger():
    logs = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(logs):
        os.makedirs(logs)
    logging.basicConfig(filename='logs/scrapper.log', level=logging.INFO)


def _validateFolders(modelName):
    predictedData = os.path.join(os.getcwd(), "predictedData")
    if not os.path.exists(predictedData):
        logging.info(f"Creating Directory: {predictedData}")
        os.makedirs(predictedData)
    temps = os.path.join(predictedData, "temps")
    if not os.path.exists(temps):
        logging.info(f"Creating Directory: {temps}")
        os.makedirs(temps)
    predictedModelData = os.path.join(os.getcwd(),"modelPrediction", modelName)
    if not os.path.exists(predictedModelData):
        logging.info(f"Creating Directory: {predictedModelData}")
        os.makedirs(predictedModelData)

def _getModel(pathToModel):
        model = load_model(pathToModel)
        return model
    
def _runThroughPictures(model, modelName):
    for root, directories, files in os.walk(os.path.join("predictedData")):
        for filename in files:
            filePath = os.path.join(root, filename)
            predictImg(filePath, model, modelName)

def predictImg(filePath, model, modelName, predictionThreashhold=0.7):
     img = cv2.imread(filePath)
     img = img /255
     img = np.expand_dims(img, axis=0)
     pred = model.predict(img, verbose=False)
     if pred > predictionThreashhold:
           saveImg(filePath, modelName)
     os.remove(filePath)

def saveImg(filePath, modelName):
      img = cv2.imread(filePath)
      tail = os.path.basename(filePath)
      savePath = os.path.join(os.getcwd(), "modelPrediction", modelName, tail)
      cv2.imwrite(savePath, img)
    
if __name__ == "__main__":
        main()