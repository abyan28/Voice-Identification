#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 14:04:20 2022

@author: shoffi
"""
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import warnings
warnings.filterwarnings('ignore')
import os,cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import librosa
from librosa.util import normalize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet import MobileNet
# from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
import pickle



import tensorflow
from tensorflow.keras.models import Model
RGB=3
DIM=224
input_shape = (DIM, DIM, RGB)
IMG_DIM = (DIM, DIM)
SET_SAMPLE_RATE = 8000

def read_audio(open_path):
    audio, sample_rate = librosa.load(open_path, SET_SAMPLE_RATE)

    print("Audio path : ",open_path)
    print("Sample rate: {0}Hz".format(sample_rate))
    print("Audio duration: {0}s".format(len(audio) / sample_rate))
    print("Audio successfully readed...\n")

    audio, index = librosa.effects.trim(audio)
    audio = librosa.effects.preemphasis(audio)
    #audio = audio[0:gv.SET_CUT*sample_rate]
    #os.remove(gv.temp_file)

    return audio, sample_rate

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
    h = array.shape[0]
    w = array.shape[1]
    a = max((xx - h) // 2,0)
    aa = max(0,xx - a - h)
    b = max(0,(yy - w) // 2)
    bb = max(yy - b - w,0)
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

def generate_features(y_cut, sr):
    try:
        max_size=1000 #my max audio file feature width
        n_fft = 255
        hop_length = 512
        n_mfcc = 128

        stft = padding(np.abs(librosa.stft(y_cut, n_fft=n_fft, hop_length = hop_length)), n_mfcc, max_size)
        MFCCs = padding(librosa.feature.mfcc(y_cut, n_fft=n_fft, hop_length=hop_length,n_mfcc=n_mfcc),n_mfcc,max_size)
        spec_centroid = librosa.feature.spectral_centroid(y=y_cut, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y_cut, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y_cut, sr=sr)

        #Now the padding part
        image = np.array([padding(normalize(spec_bw),1, max_size)]).reshape(1,max_size)
        image = np.append(image,padding(normalize(spec_centroid),1, max_size), axis=0)

        #repeat the padded spec_bw,spec_centroid and chroma stft until they are stft and MFCC-sized
        for i in range(0,9):
            image = np.append(image,padding(normalize(spec_bw),1, max_size), axis=0)
            image = np.append(image, padding(normalize(spec_centroid),1, max_size), axis=0)
            image = np.append(image, padding(normalize(chroma_stft),12, max_size), axis=0)
        image=np.dstack((image,np.abs(stft)))
        image=np.dstack((image,MFCCs))

        print("Generate feature image from audio success...\n")
    except:
        print("Generate feature image from audio failed...\n")

    return image


def read_dataset():    
    features = []
    labels   = []
    folders  = []
    dir_dataset = os.path.join(os.getcwd(), 'voice')
    print('Directory of Dataset :', dir_dataset)

    list_folder = [f for f in os.listdir(dir_dataset)]

    for f in list_folder:
        f_dir = os.path.join(dir_dataset, f)        

        for ff in os.listdir(f_dir):
            ff_path = os.path.join(f_dir,ff)

            if ff.lower().endswith('wav'):
                audio, sample_rate = read_audio(ff_path)

                audio = np.array((audio-np.min(audio))/(np.max(audio)-np.min(audio)))
                audio = audio/np.std(audio)

                feature = generate_features(audio, sample_rate)
                labels.append(f)
                folders.append(ff_path)
                features.append(feature)                          
        
    return np.array(labels),np.array(folders),np.array(features)

def createpreTrain():
    
    m = tensorflow.keras.applications.MobileNetV2(include_top=False, weights='imagenet', 
                                      input_shape=input_shape)

    
    output    = m.layers[-1].output
    output    = tensorflow.keras.layers.Flatten()(output)
    M         = Model(inputs=m.input, outputs=output)# base_model.get_layer('custom').output)

    M.trainable = False
    for layer in M.layers:
        layer.trainable = False
    return M

if __name__=="__main__":
    nmFile="dataset.npz"
    if os.path.exists(nmFile):
        print("File %s already exists"%nmFile)
        dataLoad = np.load(nmFile)
        labels  = dataLoad['labels']
        folders = dataLoad['folders']
        features = dataLoad['features']
        print("loaded success")
        
    else:
        labels,folders,features = read_dataset()
        np.savez_compressed(nmFile,labels=labels,folders=folders,features=features)
    
    # nmFile="train_test_88.npz"
    nmFile="train_test.npz"
    if os.path.exists(nmFile):
        print("File %s already exists"%nmFile)
        dataLoad = np.load(nmFile)
        x_train  = dataLoad['x_train']
        x_test   = dataLoad['x_test']
        y_train  = dataLoad['y_train']
        y_test   = dataLoad['y_test']        
        print("loaded success")
        
    else:              
        features=np.asarray(features).astype(np.float32)                 
        ftrAll=[]
        preTrain = createpreTrain()
        for i in range(len(labels)):
            img = cv2.resize(features[i],IMG_DIM) 
            img_tensor = image.img_to_array(img)                    # (height, width, channels)
            img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
            img_tensor /= 255.                 
            ftr = preTrain.predict(img_tensor)
            ftrAll.append(ftr.ravel())    
            
            
        features = np.array(ftrAll).astype(float)       
        x_train, x_test, y_train, y_test = train_test_split(features, labels, 
                                                          test_size=0.33, 
                                                          stratify=labels,
                                                           # random_state=43
                                                          )
        np.savez_compressed(nmFile,
                            x_train=x_train,
                            x_test=x_test,
                            y_train=y_train,
                            y_test=y_test,
                            )


    
    modelLR = LogisticRegression(solver='lbfgs',
                                 n_jobs=-1, 
                                  multi_class='auto',
                                  tol=0.8
                                 )
    modelLR.fit(x_train,y_train)    
    scores = modelLR.score(x_test,y_test)
    ypred  = modelLR.predict(x_test)
    with open('modelTR.pkl', 'wb') as f:
          pickle.dump(modelLR, f)
    
    
    
    print(scores)
    for i in range(len(y_test)):
        if ypred[i]!=y_test[i]:
            print("err %d ori=%s pred=%s"%(i,y_test[i],ypred[i]))
    print("test")
    audio, sample_rate  = read_audio('2022-02-28T01_22_18.082Z.wav')
    audio = np.array((audio-np.min(audio))/(np.max(audio)-np.min(audio)))
    audio = audio/np.std(audio)
    feature = generate_features(audio, sample_rate)
    img = cv2.resize(feature,IMG_DIM) 
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.   
    preTrain = createpreTrain()              
    ftr = preTrain.predict(img_tensor)
    ypred = modelLR.predict(ftr)
    print(ypred)
    
    