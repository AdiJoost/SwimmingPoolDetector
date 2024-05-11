from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, RandomFlip, RandomRotation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from dataGenerator import generateEqualDataSet
from tensorflow.keras.applications.vgg16 import VGG16
import pickle

NUMBER_OF_PICTURES = 100

BATCH_SIZE = 2
EPOCHS = 2
VALIDATION_SPLIT = 0.2
CALLBACK = None

def main():
    model = getModel()
    xTrain, xTest, yTrain, ytest = getData(NUMBER_OF_PICTURES)
    history = trainModel(model, xTrain, yTrain, batchSize=BATCH_SIZE, epochs=EPOCHS, validationSplit=VALIDATION_SPLIT, callback=CALLBACK)
    score = evaluateModel(model, xTest, ytest)
    saveModel(model, "myModel.keras", history, score)


def getModel():
    model = Sequential();
    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        ])
    model.add(data_augmentation)
    model.add(VGG16(
        include_top=False,
        input_shape=(500,500,3)
    ))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision'])
    
    return model

def getData(numberOfSamples):
    X, y = generateEqualDataSet(numberOfSamples)
    xTrain, xTest, yTrain, ytest = train_test_split(X, y, test_size=0.2)
    return xTrain, xTest, yTrain, ytest

def trainModel(model, xTrain, yTrain, batchSize, epochs, validationSplit, callback=None):
    history = model.fit(xTrain,
        yTrain,
        batch_size=batchSize,
        epochs=epochs,
        validation_split=validationSplit,
        callbacks=callback,)
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