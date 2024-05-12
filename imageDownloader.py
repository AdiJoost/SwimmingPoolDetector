import requests
import json
import logging

from pathlib import Path
import os
import time
from datetime import datetime

from swissimage_annotator.src.helpers import convert_coordinates

import matplotlib.pyplot as plt
import cv2

logger = logging.getLogger(__name__)

BASE_URL = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissimage-dop10/items"
DATA_DIR = Path("predictedData")

def main():
    for i in range(2752, 2754):
        for j in range(1212, 1215):
            gatherImages(i,j)
        
def gatherImages(x, y):
    print(f"GetImage of: {x}, {y}")
    validateFolders()
    logger.info("started")
    getImage(x, y)

def validateFolders():
    predictedData = os.path.join(os.getcwd(), "predictedData")
    if not os.path.exists(predictedData):
        logging.info(f"Creating Directory: {predictedData}")
        os.makedirs(predictedData)
    temps = os.path.join(predictedData, "temps")
    if not os.path.exists(temps):
        logging.info(f"Creating Directory: {temps}")
        os.makedirs(temps)

def getImage(x, y):
    logging.info(f"Get Image of {x}, {y}")
    download_tif(x * 1000, (x+1) * 1000, y * 1000, (y+1) * 1000, crs=2056)
    for root, directories, files in os.walk(os.path.join("predictedData", "temps")):
        for filename in files:
            filePath = os.path.join(root, filename)
            cutAndRemoveFile(filePath, f"filename")


def cutAndRemoveFile(filepath, filename):
    image = cv2.imread(filepath)
    if type(image) != None: #WTF python please... -> This Error handling is probably not going to work
        cutImages(image, filename)
    else:
        logging.critical(f"Got an empty file: {filepath}")
        return
    try:
        logging.info(f"Remove: {filepath}")
        os.remove(filepath)
    except OSError as e:
        logging.critical(f"Error: {filepath} - {e.strerror}")

def download_tif(x_min=None, x_max=None, y_min=None, y_max=None, data_dir=None, crs=4326):
    if not data_dir:
        data_dir = DATA_DIR

    if crs == 2056:
        coordinates = convert_coordinates([[x_min+100, y_min+100], [x_max-100, y_max-100]], 2056, 4326)
        x_min = coordinates[0][1]
        y_min = coordinates[0][0]
        x_max = coordinates[1][1]
        y_max = coordinates[1][0]

    query = ""
    if x_min and x_max and y_min and y_max:
        query += '?bbox=' + ','.join([str(x_min), str(y_min), str(x_max), str(y_max)])

    url = BASE_URL + query
    response = retryCall(url)
    data = json.loads(response.content.decode('utf-8'))
    for i, feature in enumerate(data['features']):
        asset = list(filter(lambda a: a['eo:gsd'] == 0.1, feature['assets'].values()))[0]
        r = retryCall(asset['href'])
        with open(f"predictedData/temps/temp_{i}", 'wb') as f:
            f.write(r.content)

def retryCall(link, delay=1, trys=15):
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json',
    }
    for _ in range(trys):
        try:
            response = requests.get(link, headers=headers)
            return response
        except:
            print(f"Retry {link}")
            time.sleep(delay)
    print(f"Failed to load after {trys} trys:\n{link}")

def cutImages(image, prefix):
    print("cutting Image")
    cuts_per_image = int(1000 / 50)
    image_width = image.shape[0]
    image_height = image.shape[1]
    cropped_width = int(image_width / cuts_per_image)
    cropped_height = int(image_height / cuts_per_image)
    for i in range(cuts_per_image):
            for j in range(cuts_per_image):
                x = i * cropped_width
                y = j * cropped_height
                new_x = 0 + int(x/10)
                new_y = 0 + int(y/10)
                croppedImage = image[x : x + cropped_width, y : y + cropped_height, :]
                timestamp = datetime.now().timestamp()
                path = os.path.join("predictedData", f'{prefix}_{new_x}_{new_y}_{timestamp}.png')
                ret = cv2.imwrite(path, croppedImage)

if __name__ == "__main__":
    main()