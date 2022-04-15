import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
with_mask = np.load('with_mask.npy')
without_mask =np.load('without_mask.npy')

with_mask = with_mask.reshape(500,80 * 80 * 3 )
without_mask = without_mask.reshape(500, 80 * 80 * 3 )

X = np.r_[with_mask, without_mask]
labels = np.zeros(X.shape[0])

labels[500:] = 1.0

names = {0 : 'Mask', 1 :'No Mask'}
x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size=0.20)

pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)

svm = SVC()
svm.fit(x_train, y_train)

x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_COMPLEX
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()
        font = cv2.FONT_HERSHEY_COMPLEX
        faces=faceDetect.detectMultiScale(frame, 1.3, 5)
        for x,y,w,h in faces:
            x1,y1=x+w, y+h
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 1)
            faces = frame[y:y+h, x:x+w, :]
            faces = cv2.resize(faces, (80,80))
            faces = faces.reshape(1, -1)
            faces = pca.transform(faces)
            pred = svm.predict(faces)[0]

            n = names[ int(pred)]
            cv2.putText(frame, n, (x,y), font, 1, (244, 250, 250), 2)
        cv2.imshow('result',frame)
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()