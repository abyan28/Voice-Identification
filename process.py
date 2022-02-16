import librosa
import tensorflow as tf
from keras import Sequential, models
from keras.applications.densenet import layers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from librosa.util import normalize

import globalvar as gv
import os
import numpy as np

import pickle
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import tkinter as tk


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



def read_audio(open_path):
    audio, sample_rate = librosa.load(open_path, gv.SET_SAMPLE_RATE)

    print("Audio path : ",open_path)
    print("Sample rate: {0}Hz".format(sample_rate))
    print("Audio duration: {0}s".format(len(audio) / sample_rate))
    print("Audio successfully readed...\n")

    audio, index = librosa.effects.trim(audio)
    audio = librosa.effects.preemphasis(audio)
    #audio = audio[0:gv.SET_CUT*sample_rate]
    #os.remove(gv.temp_file)

    return audio, sample_rate


def extract_audio(audio):
    mfcc = librosa.feature.mfcc(audio, gv.SET_SAMPLE_RATE)
    return mfcc


def read_dataset():
    dataset = []

    gv.dir_dataset = os.path.join(os.getcwd(), 'voice')
    print('Directory of Dataset :', gv.dir_dataset)

    list_folder = [f for f in os.listdir(gv.dir_dataset)]

    for f in list_folder:
        f_dir = os.path.join(gv.dir_dataset, f)

        for ff in os.listdir(f_dir):
            ff_path = os.path.join(f_dir,ff)

            if ff.lower().endswith(gv.EXT):
                audio, sample_rate = read_audio(ff_path)

                audio = np.array((audio-np.min(audio))/(np.max(audio)-np.min(audio)))
                audio = audio/np.std(audio)

                feature = generate_features(audio, sample_rate)

                dataset.append((feature, f, ff_path))

    return dataset



def set_train_data(dataset):
    dtw_dataset = []
    features=[]
    labels = []
    gv.define_labels = []
    gv.no_classes = 0

    old_label = ""
    lbl = -1
    for (feature, f, ff_path) in dataset:
        # D, wp = librosa.sequence.dtw(input, feature, metric='euclidean', subseq=True)
        # dist = D[D.shape[0] - 1, D.shape[1] - 1]
        #
        # print('Distance Value: ',dist,', for ',f,', path: ', ff_path)
        # dtw_dataset.append((dist, f, ff_path))

        features.append(feature[np.newaxis,...])

        if not (f == old_label):
            lbl = lbl + 1
            old_label = f
            gv.define_labels.append(f)
            gv.no_classes = lbl + 1

        labels.append(lbl)

    # dtw_dataset.sort()
    # return dtw_dataset
    output=np.concatenate(features,axis=0)
    return(np.array(output), np.array(labels))




def set_dataset():

    # try:
    #     gv.feature_dataset = []
    #     gv.pickle_filee = open(gv.FILE_SAVE, "rb")
    #     gv.feature_dataset = pickle.load(gv.pickle_filee)
    # except:
    #     gv.feature_dataset = []


    if True:
        try:
            print('Read All Dataset....Please Wait')
            time.sleep(3)

            feature_dataset = read_dataset()
            gv.pickle_filee = open(gv.FILE_SAVE, "wb")
            pickle.dump(feature_dataset, gv.pickle_filee)

            train_features, train_labels  = set_train_data(feature_dataset)

            #Define Model
            input_shape=(128,1000,3)
            CNNmodel = models.Sequential()
            CNNmodel.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
            CNNmodel.add(layers.MaxPooling2D((2, 2)))
            CNNmodel.add(layers.Dropout(0.2))
            CNNmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
            CNNmodel.add(layers.MaxPooling2D((2, 2)))
            CNNmodel.add(layers.Dropout(0.2))
            CNNmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
            CNNmodel.add(layers.Flatten())
            CNNmodel.add(layers.Dense(64, activation='relu'))
            CNNmodel.add(layers.Dropout(0.2))
            CNNmodel.add(layers.Dense(32, activation='relu'))
            CNNmodel.add(layers.Dense(gv.no_classes + 1, activation='softmax'))

            CNNmodel.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                             metrics=['accuracy'])
            history = CNNmodel.fit(train_features, train_labels, epochs=100)
            CNNmodel.save(gv.FILE_H5)

            history_dict=history.history
            loss_values=history_dict['loss']
            acc_values=history_dict['accuracy']
            #val_loss_values = history_dict['val_loss']
            #val_acc_values=history_dict['val_accuracy']
            epochs=range(1,101)

            root = tk.Tk()
            root.resizable(0,0)
            root.configure(background='white')

            frame = tk.Frame(root)
            frame.place(x=10, y=55)

            fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
            fig.suptitle('CNN Training Process', fontsize=16)

            ax1.plot(epochs,loss_values,'b-',label='Training Loss')
            ax1.set_title('CNN Training Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax2.plot(epochs,acc_values,'b-', label='Training Accuracy')
            ax2.set_title('CNN Training Accuracy')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            ax2.legend()

            plt.show()

            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

            width = 1200;
            height = 800;

            X = int(root.winfo_screenwidth() / 2 - width / 2)
            Y = int(root.winfo_screenheight() / 2 - height / 2)

            sizeTk = str(width) + 'x' + str(height)
            root.geometry('{}+{}+{}'.format(sizeTk, X, Y))


            root.update()
            root.mainloop()



        except Exception as e:
            print(e)

# def match_audio_from_file():
#
#     try:
#         open_path = askopenfilename(initialdir= os.getcwd(), defaultextension=".wav", title='Open an Audio File',
#                                     filetypes=(("WAVE files", ".wav"), ("All files", "*.*")))
#
#         result = match_audio(open_path)
#
#         if result == None:
#             msg.showerror('Error', 'All Process Failed...')
#
#     except Exception as e:
#         msg.showerror('Error', 'All Process Failed...')
#         print('All Process Failed...')
#         print(e)



def match_audio(open_path):
    try:
        print('Read All Dataset....Please Wait')
        time.sleep(3)

        feature_dataset = read_dataset()
        gv.pickle_filee = open(gv.FILE_SAVE, "wb")
        pickle.dump(feature_dataset, gv.pickle_filee)

        train_features, train_labels  = set_train_data(feature_dataset)

        audio, sample_rate = read_audio(open_path)

        print('Read Test Audio Data for Identification:')
        features=[]
        feature = generate_features(audio, sample_rate)
        features.append(feature[np.newaxis,...])
        features=np.concatenate(features,axis=0)

        labels = []
        labels.append(1)

        test_features, test_labels = (np.array(features), np.array(labels))

        CNNmodel = models.load_model(gv.FILE_H5)

        prediction = CNNmodel.predict(test_features)
        cla = np.argmax(prediction)
        val = round(prediction[0,cla] * 100,4)
        match = gv.define_labels[cla]

        print('\n')
        print('Result:\n','-------------------------------\n',
        'This voice identified as ',match.upper(),'\n with probability match = ',str(val),'%')

        return str(match.upper() + '    (with probability match = ' + str(val) + '%)')


    except Exception as e:
        print(e)

        return gv.EMPTY



