import os
from tkinter import *

from tkinter.filedialog import askopenfilename
import tkinter.messagebox as msg
from librosa.util import normalize

import librosa
import numpy as np

from PIL import ImageTk,Image


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
    return image


def read_audio_from_file():

    try:
        open_path = askopenfilename(initialdir= os.getcwd(), defaultextension=".wav", title='Open an Audio File',
                                    filetypes=(("WAVE files", ".wav"), ("All files", "*.*")))

        y,sr=librosa.load(open_path, sr=44100) #load the file
        img_np = generate_features(y, sr)

        image = Image.fromarray(img_np, 'RGB')
        image.save('temp.png')
        image.show()


    except Exception as e:
        msg.showerror('Error', 'All Process Failed...')
        print('All Process Failed...')
        print(e)


if __name__ == '__main__':
    read_audio_from_file()