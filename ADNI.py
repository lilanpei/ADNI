
# coding: utf-8

# In[ ]:


import os
import numpy as np
import nibabel as nib
import glob
import math
import random
import time
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import keras
n_folds = 5
batch_size = 5
n_epochs = 150
save_dir = os.path.join(os.getcwd(), 'saved_models')
ds_path = '/home/lilanpei/work/ADNI'
historyloglocation_train  = './{}_traininghistory_train.txt'.format(str(time.time()))
historyloglocation_test  = './{}_traininghistory_train.txt'.format(str(time.time()))
model_name = 'ADNI_VoxCNN'
lr = 27*1e-6
ds_name = [["AD","NC"],["AD","EMCI"],["AD","LMCI"],["LMCI","NC"],["LMCI","EMCI"],["EMCI","NC"]]

def get_model(summary=False):
    """ Return the Keras model of the network
    """
    model = Sequential()
    # 1st Volumetric Convolutional block
    model.add(Convolution3D(8, (3, 3, 3), activation='relu', padding='same', input_shape=(110, 110, 110, 1)))
    model.add(Convolution3D(8, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # 2nd Volumetric Convolutional block
    model.add(Convolution3D(16, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(16, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # 3rd Volumetric Convolutional block
    model.add(Convolution3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # 4th Volumetric Convolutional block
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    # 1th Deconvolutional layer with batchnorm and dropout for regularization
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    # 2th Deconvolutional layer
    model.add(Dense(64, activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.7))
    # Output with softmax nonlinearity for classification
    model.add(Dense(2, activation='softmax'))
    if summary:
        print(model.summary())
    return model

def get_data_set(ds_name_1,ds_name_2):
    ds_1 = list()
    ds_2 = list()
    lb = list()

    ds_1_path = "{}/{}".format(ds_path, ds_name_1)
    ds_2_path = "{}/{}".format(ds_path, ds_name_2)
    #print(ds_1_path)
    #print(ds_2_path)
    
    for path, dirs, files in os.walk(ds_1_path):
        for d in dirs:
            for path in glob.iglob(os.path.join(path, d, '*.nii')):
                image = nib.load(path)
                img = image.get_fdata()
                ds_1.append(img)
                lb.append([1,0])
         
    for path, dirs, files in os.walk(ds_2_path):
        for d in dirs:
            for path in glob.iglob(os.path.join(path, d, '*.nii')):
                image = nib.load(path)
                img = image.get_fdata()
                ds_2.append(img)
                lb.append([0,1])
                
    ds_1 = np.asarray(ds_1)
    ds_2 = np.asarray(ds_2)
    ds_1 = np.reshape(ds_1,(len(ds_1),ds_1[0].shape[0],ds_1[0].shape[1],ds_1[0].shape[2],1))
    ds_2 = np.reshape(ds_2,(len(ds_2),ds_2[0].shape[0],ds_2[0].shape[1],ds_2[0].shape[2],1))
    lb = np.asarray(lb)

    return ds_1, ds_2, lb

def gen_data(data,label):
    print ("in single generator")
    while True:
        indices = list(range(len(data)))
        random.shuffle(indices)
        for i in indices:
            x = data[i]
            y = label[i]
            yield x, y

def gen_folds(data_1,data_2,label):
    print("in flods generator")
    indices_1 = list(range(len(data_1)))
    indices_2 = list(range(len(data_1),len(data_1)+len(data_2)))
    indices_folds = list(range(n_folds))
    folds_1 = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1).split(data_1,label[:len(data_1),0]))
    folds_2 = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1).split(data_2,label[len(data_1):len(data_1)+len(data_2),0]))
    while True:
        for i in indices_folds:
            x_train = np.concatenate((data_1[folds_1[i][0]],data_2[folds_2[i][0]]),axis=0)
            y_train = np.concatenate((label[folds_1[i][0]],label[folds_2[i][0]+len(data_1)]),axis=0)
            x_test = np.concatenate((data_1[folds_1[i][1]],data_2[folds_2[i][1]]),axis=0)
            y_test = np.concatenate((label[folds_1[i][1]],label[folds_2[i][1]+len(data_1)]),axis=0)
            yield x_train,y_train,x_test,y_test

def gen_batch(data,label):
    print ("in batch generator")
    counter = Counter([tuple(ele) for ele in label])
    len_1 = list(counter.items())[0][1]
    len_2 = list(counter.items())[1][1]
    #print(counter.elements)
    data_1_gen = gen_data(data[:len_1,],label[:len_1,])
    data_2_gen = gen_data(data[len_1:len_1+len_2,],label[len_1:len_1+len_2,])
    #while True:
    x_batch = []
    y_batch = []
    x_1, y_1 = next(data_1_gen)
    x_batch.append(x_1)
    y_batch.append(y_1)
    x_2, y_2 = next(data_2_gen)
    x_batch.append(x_2)
    y_batch.append(y_2)
    for _ in range(batch_size-len(x_batch)):
        if random.randint(1,100) <= 50:
            x_3, y_3 = next(data_1_gen)
            x_batch.append(x_3)
            y_batch.append(y_3)
        else:
            x_4, y_4 = next(data_2_gen)
            x_batch.append(x_4)
            y_batch.append(y_4)
                
    yield np.array(x_batch),np.asarray(y_batch)

def one_vs_one_train(ds_name_1,ds_name_2):
    ds_1, ds_2, lb = get_data_set(ds_name_1,ds_name_2)
    folds_gen = gen_folds(ds_1,ds_2,lb)
    model = get_model(summary=True)
    opt = keras.optimizers.Adam(lr)
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
    for x_train,y_train,x_test,y_test in folds_gen:
        batch_gen = gen_batch(x_train,y_train)
        for i in range(n_epochs):
            for x_batch, y_batch in batch_gen:
                history_train = model.fit(x_batch, y_batch, epochs=1)
                print ('epoch: ', i)
                print ('loss: ',history_train.history['loss'][0])
                with open(historyloglocation_train,"a+") as f:
                    f.write('{},{},{}\n'.format(str(i), str(history_train.history['loss'][0]), str(history_train.history['acc'][0])))
            history_eval = model.evaluate(x_test,y_test)
            print(history_eval.history['acc'])
            print(history_eval.history['loss'])
            with open(historyloglocation_test,"a+") as f:
                f.write('{},{},{}\n'.format(str(i), str(history_eval.history['loss'][0]), str(history_train.history['acc'][0])))
            #model.save_weights(save_dir)
            #model.save(save_dir)    
for i in range(len(ds_name)):
    one_vs_one_train(str(ds_name[i][0]),str(ds_name[i][1]))


# In[11]:


import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

