import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime


path='D:facerecog'
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)


for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
        with open('D:attendance.csv', 'r+')as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
        

# date=now.strftime("%x")
#             dtString=now.strftime('%H:%M:%S')
#             dtweek=now.strftime("%A")

encodeListKnow=findEncodings(images)
print("encoding complete")
      
      
      

cap = cv2.VideoCapture(1)

while True:
      success,img=cap.read()
      imgs=cv2.resize(img,(0,0),None,0.25,0.25)
      imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
      
      facesCurFrame=fr.face_locations(imgs)

      encodecurFrame=fr.face_encodings(imgs,facesCurFrame)
      
      
      
      
      
      for encodeFace,faceLoc in zip(encodecurFrame,facesCurFrame):
           matches=fr.compare_faces(encodeListKnow,encodeFace)
           faceDis=fr.face_distance(encodeListKnow,encodeFace)
#            print(faceDis)
      
           matchIndex=np.argmin(faceDis)
        
           if matches[matchIndex]:
                name=classNames[matchIndex].upper()
                print(name)
                y1,x2,y2,x1=faceLoc
                y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                markAttendance(name)
                
      cv2.imshow('Webcam',img)
      cv2.waitKey(1)