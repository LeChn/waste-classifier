import csv
from PIL import Image
import numpy as np
import pdb
import cv2
import pandas as pd
import os
from matplotlib import pyplot as plt
import xml.etree.ElementTree

def load_csv_file(filename):
    pread = pd.read_csv('data.csv')
    # imgNames = pread.
    labels = pread.BinaryResidential
    imgNames = pread["Photo Filename"].values.reshape(len(labels), 1)
    labels = pread["BinaryResidential"].values.reshape(len(labels), 1)
    labels = np.hstack((imgNames,labels))
    dictionary = dict()
    for combo in labels:
        dictionary[combo[0] + '.JPG'] = combo[1]
    return dictionary

def prepareTraining(X, y, im, label, numGrids, file):
    imgwidth = len(blur[0])
    width = int(len(blur[0])/numGrids)
    imgheight = len(blur)
    height = int(len(blur)/numGrids)
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            grid = im[i:i+height, j:j+width,:].reshape(1, height*width*3)
            X.append(grid.tolist())
            print("Added one sample " + file + " i " + str(i) + " j " + str(j))
    y = np.vstack((y, np.ones((numGrids * numGrids, 1)) * label))
    return (X,y)

# if __name__ == '__main__':
size = 1152
numGrids = 12

print("Height of individual grid", size/numGrids)
print("Width of individual grid", size/numGrids*1.5)

X = []
y = np.zeros((0,1)) 

labels = load_csv_file("data.csv")

print("Finished importing")

maxNumPix = 30

info = [file for file in os.listdir(".") if file[-3:]=="JPG"]

for file in os.listdir("."):
    if(file.endswith(".JPG")):
        img = cv2.imread(file)
        if (os.path.isfile(file[0:-3] + "xml")):
            e = xml.etree.ElementTree.parse(file[0:-3] + "xml").getroot()
        else:
            continue

        bndbox = e[6][4]
        xmin = int(bndbox[0].text)
        ymin = int(bndbox[1].text)
        xmax = int(bndbox[2].text)
        ymax = int(bndbox[3].text)
        fullWidth = xmax - xmin
        fullHeight = ymax - ymin

        img = img[ymin:ymax, xmin:xmax, :]

        blur = cv2.resize(img, (int(224 * numGrids),int(224 * numGrids)))
        # blur = cv2.resize(img, (int(size*1.5),size)).reshape(len(blur)*len(blur[0])*len(blur[0][0]),1)
        (X, y) = prepareTraining(X, y, blur, labels[file], numGrids, file)
        maxNumPix -= 1
    if maxNumPix <= 0:
        break
X = np.asarray(X).reshape(-1, int(224*224*3))
np.save('Xmatrix.npy', X)
np.save('Ylabels.npy', y)