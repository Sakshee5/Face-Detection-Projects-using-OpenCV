import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

path = 'C:/Users/Sakshee/Pictures/Image_attendance'
images = []       # List with all images
className = []    # List containing corresponding class Names

myList = os.listdir(path)
# print(myList)
# prints the names of the 3 images in the folder
print("Total Classes Detected:",len(myList))

for cls in myList:
        curImg = cv2.imread(f'{path}/{cls}')
        images.append(curImg)
        className.append(os.path.splitext(cls)[0])

# print(className)

def findEncodings(images):
    encodeList = []        # to attach encodings of all images in our given folder
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    # 'r+' means read and write at the same time
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        # print(myDataList)
        # prints ['Name', 'Time']

        # to append names found
        nameList =[]

        # since ['Name', 'Time'] is one line and later we'll have multiple lines
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dt_string}')

encodeListKnown = findEncodings(images)
# print(len(encodeListKnown)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    # resizing to speed up the process
    imgSmall = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgSmall)
    encodesCurrentFrame = face_recognition.face_encodings(imgSmall, facesCurrentFrame)

    # comparing the live footage with the images database to find matches
    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # print(faceDis)
        # gives a list of 3 values

        matchIndex = np.argmin(faceDis)
        # returns the index of min. faceDis so that we know which image has the highest resemblance

        if faceDis[matchIndex] < 0.50:
            name = className[matchIndex].upper()
            markAttendance(name)
        else:
            name = 'Unknown'
        # print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)