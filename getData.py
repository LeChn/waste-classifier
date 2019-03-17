import csv
from PIL import Image
import numpy as np
import pdb
import cv2
import pandas as pd
import os
from matplotlib import pyplot as plt
import json
from pprint import pprint


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
    previousXlen = len(X)
    imgwidth = len(blur[0])
    width = int(len(blur[0])/numGrids)
    imgheight = len(blur)
    height = int(len(blur)/numGrids)
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            grid = im[i:i+height, j:j+width,:].reshape(1, height*width*3)
            X = np.vstack((X, grid))

            print("Added one sample " + file + " i " + str(i) + " j " + str(j))

    y = np.vstack((y, np.ones((len(X) - previousXlen, 1)) * label))
    return (X,y)

if __name__ == '__main__':
    
    size = 72
    numGrids = 12

    X = np.zeros((0,int(size * size * 1.5 * 3 / numGrids**2)))
    y = np.zeros((0,1)) 

    labels = load_csv_file("data.csv")

    print("Finished importing")

    maxNumPix = 30
for file in os.listdir("./imageLabel/."):
    if(file.endswith(")")):
        print("Extracting labels from file " + file)
        data = json.loads(open('./imageLabel/' + file + "/data_file.json").read())
        img = cv2.imread("../../East Team S2 2/" + file + ".jpg")
        if 'labels' in data:
            height, width, channels = img.shape
            gheight, gwidth = int(height/numGrids) , int(width/numGrids)
            for index, gridLabel in enumerate(data['labels']):
                if gridLabel:
                    rowIndex = index % numGrids
                    colIndex = int(index / numGrids)
                    grid = img[rowIndex * gheight : (rowIndex + 1) * gheight, colIndex * gwidth : (colIndex + 1) * gwidth, :]
                    blur = cv2.resize(img, (256, 256))



    # np.save('Xmatrix.npy', X)
    # np.save('Ylabels.npy', y)


    # X = np.load('Xmatrix.npy')
    # y = np.load('Ymatrix.npy')

    pdb.set_trace()

    xTx = X.T.dot(X)
    xTx_inv = np.linalg.inv(xTx)
    weight = xTx_inv.dot(X.T.dot(y))
    pdb.set_trace()


    y_hat = X.dot(weight)
    counter = 0
    for i, eachy in enumerate(y_hat):
        if(int(eachy + 0.5) == int(y[i])):
            counter += 1
    # weight = np.load('weight.npy')
    pdb.set_trace()
    print("Accurracy is " + str(counter/len(y)) + "%")
