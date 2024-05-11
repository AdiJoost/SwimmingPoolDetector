from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, RandomFlip, RandomRotation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from dataGenerator import generateEqualDataSet
from tensorflow.keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import pickle

NUMBER_OF_PICTURES = 114

NODES_AFTER_BASE_MODEL = 64

BATCH_SIZE = 6
EPOCHS = 2
CALLBACK = [ModelCheckpoint("vgg16.h5", monitor='accuracy', verbose=1, save_best_only=True, mode='max'),]
METRICS =["accuracy"]

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
    xTrain, xTest, yTrain, ytest = getData(NUMBER_OF_PICTURES)
    datagen = getDataGenerator(xTrain)
    history = trainModel(model, xTrain, yTrain, datagen, batchSize=BATCH_SIZE, epochs=EPOCHS, callback=CALLBACK)
    score = evaluateModel(model, xTest, ytest)
    saveModel(model, "lastModel.h5", history, score)


def getModel():
    baseModel = VGG16(include_top=False, input_shape=(500,500,3))
    baseModel.trainable = False

    model = Sequential([
        baseModel,
        Flatten(),
        Dense(NODES_AFTER_BASE_MODEL, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    optimizer = getOptimizer()

    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
              metrics=METRICS)
    
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

def getData(numberOfSamples):
    X, y = generateEqualDataSet(numberOfSamples)
    xTrain, xTest, yTrain, ytest = train_test_split(X, y, test_size=0.2)
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
    history = model.fit(
        datagen.flow(xTrain, yTrain, batch_size=batchSize),
        steps_per_epoch= len(xTrain) / batchSize,
        epochs=epochs,
        callbacks=callback)
    return history

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