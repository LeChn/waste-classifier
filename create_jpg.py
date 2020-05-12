from PIL import Image
import numpy as np
import os
import cv2
import json

numGrids = 12
features = []
y = []
count = 0
for file in os.listdir("./imageLabel/."):
    if(file.startswith("A")):
        print("Making photos from file " + file)
        data = json.loads(open('imageLabel/' + file + "/data_file.json").read())
        imgPath = "./imagesDataset/" + file + ".jpg"
        if os.path.isfile(imgPath) and 'labels' in data:
            img = cv2.imread("imagesDataset/" + file + ".jpg")
            height, width, channels = img.shape
            gheight, gwidth = int(height/numGrids), int(width/numGrids)
            for index, gridLabel in enumerate(data['labels']):
                if gridLabel:
                    colIndex = index % numGrids
                    rowIndex = int(index / numGrids)
                    grid = img[rowIndex * gheight: (rowIndex + 1) * gheight, colIndex * gwidth: (colIndex + 1) * gwidth, :]
                    data = grid
                    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
                    im = Image.fromarray(rescaled)
                    im.save(f'./data/{file[3:-1]}_{index}.jpg')
