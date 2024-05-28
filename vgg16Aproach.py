from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, RandomFlip, RandomRotation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, Accuracy
from sklearn.model_selection import train_test_split
from dataGenerator import generateEqualDataSet
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.initializers import Constant
from keras.preprocessing.image import ImageDataGenerator
import pickle

#Load Data Parameters
NUMBER_OF_PICTURES = 231
TRUE_FALSE_RATIO = 0.1
#Model Parameters
NODES_AFTER_BASE_MODEL = 64

#TrainingParameters
BATCH_SIZE = 36
EPOCHS = 50
EPOCHS_AFTER_UNFREEZING = 20
CALLBACK = [ModelCheckpoint("vgg16_V5.h5", monitor='accuracy', verbose=1, save_best_only=True, mode='max'),]
METRICS =[Accuracy(), TruePositives(), TrueNegatives(), FalsePositives()]
TEST_SIZE = 0.2

#Optimizer
INITIAL_LEARNING_RATE = 0.001
DECAY_STEPS = 1000
DECAY_RATE = 0.99
DECAY_STAIRCASE = True

#CALLBACKS / NOT IMPLEMENTED
PATIENCE = 10

#Datagen
ROTATION_RANGE = 40
WIDTH_SHIFT_RANGE = 50
HEIGTH_SHIFT_RANGE = 50
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True

def main():
    model = getModel()
    xTrain, xTest, yTrain, ytest = getData(NUMBER_OF_PICTURES, TRUE_FALSE_RATIO)
    datagen = getDataGenerator(xTrain)
    history = trainModel(model, xTrain, yTrain, datagen, batchSize=BATCH_SIZE, epochs=EPOCHS, callback=CALLBACK)
    for layer in model.layers[:-4]:
        layer.trainable = False
    for layer in model.layers[-4:]:
        layer.trainable = True
    trainModel(model, xTrain, yTrain, datagen, batchSize=BATCH_SIZE, epochs=EPOCHS_AFTER_UNFREEZING, callback=CALLBACK)

    score = evaluateModel(model, xTest, ytest)
    saveModel(model, "lastModel.h5", history, score)


def getModel():
    baseModel = VGG16(include_top=False, input_shape=(500,500,3))
    baseModel.trainable = False

    outputBias = Constant(TRUE_FALSE_RATIO)

    model = Sequential([
        baseModel,
        Flatten(),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid', bias_initializer=outputBias)
    ])

    optimizer = getOptimizer()

    model.compile(optimizer=optimizer,
                loss= BinaryCrossentropy(),
              metrics=METRICS,)
    
    return model

def getOptimizer():
    decay = getDecay()
    return Adam(learning_rate=decay)

def getDecay():
    return ExponentialDecay(
        INITIAL_LEARNING_RATE,
        DECAY_STEPS,
        DECAY_RATE,
        staircase=DECAY_STAIRCASE
    )

def getData(numberOfSamples, trueFalseRatio):
    X, y = generateEqualDataSet(numberOfSamples, trueFalseRatio)
    xTrain, xTest, yTrain, ytest = train_test_split(X, y, test_size=TEST_SIZE, stratify=y)
    return xTrain, xTest, yTrain, ytest

def getDataGenerator(xTrain):
    datagen = ImageDataGenerator(
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGTH_SHIFT_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        vertical_flip=VERTICAL_FLIP
    )
    datagen.fit(xTrain)
    return datagen

def trainModel(model, xTrain, yTrain, datagen, batchSize, epochs, callback=None):
    classWeights = _getClassWeigths(xTrain)
    history = model.fit(
        datagen.flow(xTrain, yTrain, batch_size=batchSize),
        steps_per_epoch= len(xTrain) / batchSize,
        epochs=epochs,
        callbacks=callback,
        class_weight = classWeights)
    return history

def _getClassWeigths(xTrain):
    notPoolWeight = (1 / (1 - TRUE_FALSE_RATIO)) * (len(xTrain) / 2)
    poolWeight = (1 / (TRUE_FALSE_RATIO)) * (len(xTrain) / 2)
    return {0: notPoolWeight, 1: poolWeight}

def evaluateModel(model, xTest, yTest):
    score = model.evaluate(xTest, yTest, verbose=0)
    return score

def saveModel(model, path, history=None, score=None):
    model.save(path)
    if history:
        with open('modelHistory.pkl', 'wb') as file:
            pickle.dump(history.history, file)
    if score:
        with open("metaData.json", "w", encoding="utf-8") as file:
            file.write(f'"score": {score}')



if __name__ == "__main__":
    main()