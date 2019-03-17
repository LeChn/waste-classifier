
# coding: utf-8

# In[50]:

import tensorflow as tf


# In[51]:

import numpy as np


# In[52]:

train_X = np.load('Xmatrix.npy')
train_Y = np.load('Ylabels.npy')


# In[53]:

from keras.utils import to_categorical
import matplotlib.pyplot as plt


# In[54]:

train_X = np.load('Xmatrix.npy')
train_Y = np.load('Ylabels.npy')
print('Training data shape : ', train_X.shape, train_Y.shape)


# In[55]:

102*144


# In[56]:

classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)


# In[67]:

height = 24
width = int(height * 1.5)

train_X = train_X.reshape(-1, height,width, 3)
# test_X = test_X.reshape(-1, 24,36, 3)
train_X.shape

train_X = train_X.astype('float32')

train_X = train_X / 255.

train_Y_one_hot = to_categorical(train_Y)
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])


# In[68]:

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[5000,:,:])
plt.title("Ground Truth : {}".format(train_Y[0]))


# In[62]:

from sklearn.model_selection import train_test_split

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)


# In[69]:

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


# In[73]:

batch_size = 64
epochs = 20
num_classes = 2

johnnychan_model = Sequential()
johnnychan_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(height,width,3)))
johnnychan_model.add(LeakyReLU(alpha=0.1))
johnnychan_model.add(MaxPooling2D((2, 2),padding='same'))
johnnychan_model.add(Dropout(0.25))
johnnychan_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
johnnychan_model.add(LeakyReLU(alpha=0.1))
johnnychan_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
johnnychan_model.add(Dropout(0.25))
johnnychan_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
johnnychan_model.add(LeakyReLU(alpha=0.1))                  
johnnychan_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
johnnychan_model.add(Dropout(0.4))
johnnychan_model.add(Flatten())
johnnychan_model.add(Dense(128, activation='linear'))
johnnychan_model.add(LeakyReLU(alpha=0.1))           
johnnychan_model.add(Dropout(0.3))
johnnychan_model.add(Dense(num_classes, activation='softmax'))


# In[74]:

johnnychan_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# In[ ]:

waste_train = johnnychan_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))


# In[ ]:

johnnychan_model.summary()


# In[ ]:

# johnnychan_model.save("johnnychan_model_dropout.h5py")


# In[ ]:

from keras.models import load_model
johnnychan_model = load_model('johnnychan_model_dropout.h5py')


# In[ ]:

train_X = np.load('Xmatrix.npy')
train_Y = np.load('Ylabels.npy')

train_X = train_X.reshape(-1, height, width, 3)
# test_X = test_X.reshape(-1, 24,36, 3)
train_X.shape

train_X = train_X.astype('float32')


train_X = train_X / 255.


# In[ ]:

train_X.shape


# In[ ]:

k = johnnychan_model.predict_classes(train_X)
train_Y = np.load('Ylabels.npy')


# In[ ]:

numGrids = 12**2
counter = 0
for i in range (0, int(len(k) / numGrids)):
    votes = k[numGrids * i: numGrids * (i + 1)]
    if int(sum(votes)* 2 / numGrids) == train_Y[i * numGrids]:
        counter += 1
counter        


# In[ ]:

numGrids = 12**2

for i in range (0, int(len(k) / numGrids)):
    votes = k[numGrids * i: numGrids * (i + 1)]
    #if sum(votes) > numGrids / 2:
        #print(i, " Residential")
    #else:
        #print(i, " Nonresidential")


# In[ ]:

#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[ ]:

def reconstructPic(X, numGrids):  
    reconstruction = np.zeros((len(X[0]) * numGrids, (len(X[0][0])) * numGrids, 3))
    width = len(X[0][0])
    imgwidth = int(len(X[0][0]) * numGrids)
    height = len(X[0])
    imgheight = int(len(X[0]) * numGrids)
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            result = X[int(numGrids*i/len(X[0])) +int(j/len(X[0][0]))].reshape(height, width, -1)
            reconstruction[i:i+height, j:j+width,:] = result
    return reconstruction


# In[ ]:

train_X[0:144].shape


# In[ ]:

def voteMap(X, k, numGrids, job, graf):
    vmap = np.zeros((len(X[0]) * numGrids, (len(X[0][0])) * numGrids, 3))
    width = len(X[0][0])
    imgwidth = int(len(X[0][0]) * numGrids)
    height = len(X[0])
    imgheight = int(len(X[0]) * numGrids)
    cnt = 1
    if(graf):
        fig=plt.figure(figsize=(7.5, 5))
        plt.title('Votes')
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            colorVote = X[int(numGrids*i/len(X[0])) +int(j/len(X[0][0]))].reshape(height, width, -1)
            if job > 0:
                if k[int(numGrids*i/len(X[0])) +int(j/len(X[0][0]))] == 0:
                    colorVote[:] = (1, 0, 0)
            else:
                if k[int(numGrids*i/len(X[0])) +int(j/len(X[0][0]))] > 0:
                    colorVote[:] = (1, 0, 0)
            vmap[i:i+height, j:j+width,:] = colorVote
            if(graf):
                temp = vmap[i:i+height, j:j+width,:]
                ax = fig.add_subplot(rows, columns, cnt)
                cnt += 1
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(temp)
    return vmap


# In[ ]:

def plotReconstruction(id):
    numGrids = 12
    X = np.copy(train_X[id*numGrids**2:(id+1)*numGrids**2])
    fig=plt.figure(figsize=(15, 10))
    plt.subplot(121), plt.imshow(reconstructPic(X, numGrids)), plt.title("Original")
    plt.subplot(122), plt.imshow(voteMap(X, k[id*numGrids**2:(id+1)*numGrids**2], numGrids, train_Y[id * numGrids**2], False)), plt.title("Votes")
    print(sum(k[id*numGrids**2:(id+1)*numGrids**2] == train_Y[id * numGrids**2]), "/", numGrids**2, "correctly identified")
    print(sum(k[id*numGrids**2:(id+1)*numGrids**2] == train_Y[id * numGrids**2])/numGrids**2*100, "% Corresponding accuracy")


# In[ ]:

plotReconstruction(40)


# In[ ]:

johnnychan_model


# In[ ]:

accuracy = waste_train.history['acc']
val_accuracy = waste_train.history['val_acc']
loss = waste_train.history['loss']
val_loss = waste_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()
plt.show()


# In[ ]:

def plotVotes(id):
    X = np.copy(train_X[id*144:(id+1)*144])
    allVote = voteMap(X, k[id*144:(id+1)*144], 12, train_Y[id * 144], False)
    fig=plt.figure(figsize=(7.5, 5))
    columns = 12
    rows = 12
    plt.title('Votes')
    for i in range(1, columns*rows +1):
        row = int((i - 1)/rows)
        col = int((i - 1)%rows)
        img = allVote[row*24:row*24+24,col*36:col*36+36,:]
        ax = fig.add_subplot(rows, columns, i)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img)
    plt.subplots_adjust(wspace=0.01, hspace=0.1)
    plt.savefig('example.png')
    plt.show()
plotVotes(40)


# In[ ]:

import scipy


# In[ ]:

from ipywidgets import interact, widgets
from scipy import stats


# In[ ]:

train_X = np.load('Xmatrix.npy')
train_Y = np.load('Ylabels.npy')

train_X = train_X.reshape(-1, 24,36, 3)
# test_X = test_X.reshape(-1, 24,36, 3)
train_X.shape

train_X = train_X.astype('float32')


train_X = train_X / 255.


# In[ ]:

interact(lambda lam: plotReconstruction(lam), lam=(0, 102))


# In[ ]:

X = train_X


# In[ ]:

from keras import backend as K
get_3rd_layer_output = K.function([johnnychan_model.layers[0].input],
                                  [johnnychan_model.layers[2].output])


# In[ ]:

johnnychan_model.layers[0]


# In[ ]:

X[3000].shape
plt.imshow(X[2510])
plt.show()


# In[ ]:

layer_output = get_3rd_layer_output([X[3000:3001]])[0]
layer_output.shape


# In[ ]:

johnnychan_model.layers


# In[ ]:

import tensorflow as tf
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
batch_size = 64
epochs = 40
num_classes = 2

johnnychan_model = Sequential()
johnnychan_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(24,36,3)))
johnnychan_model.add(LeakyReLU(alpha=0.1))
johnnychan_model.add(MaxPooling2D((2, 2),padding='same'))
johnnychan_model.add(Dropout(0.25))
johnnychan_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
johnnychan_model.add(LeakyReLU(alpha=0.1))
johnnychan_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
johnnychan_model.add(Dropout(0.25))
johnnychan_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
johnnychan_model.add(LeakyReLU(alpha=0.1))                  
johnnychan_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
johnnychan_model.add(Dropout(0.4))
johnnychan_model.add(Flatten())
johnnychan_model.add(Dense(128, activation='linear'))
johnnychan_model.add(LeakyReLU(alpha=0.1))           
johnnychan_model.add(Dropout(0.3))
johnnychan_model.add(Dense(num_classes, activation='softmax'))
johnnychan_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

from keras.models import load_model
johnnychan_model = load_model('C:\\Users\\USER\\Desktop\\johnnychan_model_dropout.h5py')
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:

layer_dict = dict([(layer.name, layer) for layer in model.layers])

