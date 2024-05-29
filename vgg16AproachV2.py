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
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import keras
import pickle



import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa



#Load Data Parameters
NUMBER_OF_PICTURES = 116
TRUE_FALSE_RATIO = 0.1
#Model Parameters
NODES_AFTER_BASE_MODEL = 64

#TrainingParameters
BATCH_SIZE = 36
EPOCHS = 50
EPOCHS_AFTER_UNFREEZING = 20
CALLBACK = [ModelCheckpoint("vgg16_V6.keras", monitor='accuracy', verbose=1, save_best_only=True, mode='max'),]
METRICS = [Accuracy(), TruePositives(), TrueNegatives(), FalsePositives()]
TEST_SIZE = 0.2

#Optimizer
INITIAL_LEARNING_RATE = 0.01
DECAY_STEPS = 1000
DECAY_RATE = 0.99
DECAY_STAIRCASE = True

#CALLBACKS / NOT IMPLEMENTED
PATIENCE = 4

#Datagen
ROTATION_RANGE = 90
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
                loss= keras.losses.BinaryFocalCrossentropy(alpha=TRUE_FALSE_RATIO, apply_class_balancing=True),
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
    
    seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
    ], random_order=True) # apply augmenters in random order
    
    ig = ImageDataGenerator(preprocessing_function=seq.augment_image)  # pass this as the preprocessing function
    
    # gen = ig.flow_from_directory(xTrain)  # nothing else changes with the generator
    
    ig.fit(xTrain)
    return ig

    
def trainModel(model, xTrain, yTrain, datagen, batchSize, epochs, callback=None):
    classWeights = _getClassWeigths(xTrain)
    history = model.fit(
        datagen.flow(xTrain, yTrain, batch_size=batchSize),
        steps_per_epoch= int(len(xTrain) / batchSize),
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