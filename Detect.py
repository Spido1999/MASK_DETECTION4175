import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

with_mask = np.load('with_mask.npy')
without_mask =np.load('without_mask.npy')

with_mask = with_mask.reshape(500,80 * 80 * 3 )
without_mask = without_mask.reshape(500, 80 * 80 * 3 )

X = np.r_[with_mask, without_mask]
labels = np.zeros(X.shape[0])

labels[500:] = 1.0

names = {0 : 'Mask', 1 :'No Mask'}

x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size=0.20)
#x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size=0.25)
#x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size=0.30)

pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)

svm = SVC()
svm.fit(x_train, y_train)

x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)

accu=accuracy_score(y_test, y_pred)
print("Accuracy is :",accu*100)


mae = mean_absolute_error(y_pred,y_test)
print("Mean Absolute Error is :" ,mae*100)
#haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#address= "http://192.168.43.1:8080/video"
#capture.open(address)
#capture = cv2.VideoCapture(0)
#data = []
#font = cv2.FONT_HERSHEY_COMPLEX
#while True:
 #   flag, img = capture.read()
  #  if flag:
   #     face = haar_data.detectMultiScale(img)
    #    for x,y,w,h in face:
     #       cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 222), 4)
      #      faces = img[y:y+h, x:x+w, :]
       #     faces = cv2.resize(faces, (80,80))
        #    faces = faces.reshape(1, -1)
         #   faces = pca.transform(faces)
          #  pred = svm.predict(faces)[0]
           # n = names[ int(pred)]
            #cv2.putText(img, n, (x,y), font, 1, (244, 250, 250), 2)
            #print(n)
        #cv2.imshow('result',img)
        #if cv2.waitKey(2) == 27:
         #   break
#capture.release()
#cv2.destroyAllWindows()
        