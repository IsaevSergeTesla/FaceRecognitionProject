import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

print('Encoding started')
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
#print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])



def findEncodings(images):
    encodeList = []
    i = 1
    imagesLen = len(images)
    for img in images:
        print(i, '/', imagesLen)
        i = i + 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    print('End of images list')
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        #if name not in nameList:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding completed')

cap = cv2.VideoCapture(0)

while(True):
    success, img = cap.read()
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#    cv2.imshow('Video', img)
#   if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            #если зафиксирован уже известный
            #формируем имя для вывода на фото и в лог
            name = classNames[matchIndex].upper()
        else:
            #если зафиксирован неизвестный
            #формируем гарантированно уникальное имя для вывода на фото, в лог и создания файла с этим именем
            now = datetime.now()
            dtString = now.strftime('%y%m%d%H%M%S')
            name = 'Unknown_' + now.strftime('%f_') + dtString
            #сохраняем img в каталог с картинками
            isWritten = cv2.imwrite(f'{path}/{name}.jpg', img)
            if isWritten:
                print('Image ', name, ' is successfully saved as file.')
            else:
                print('Image ', name, ' is successfully did not save as file.')
            #добавляем encode к спиcку известных , чтобы повторно не зафиксировать неизвестного
            encodeListKnown.append(encodeFace)
            #Добавляем имя к спсику имен
            classNames.append(name)

        print(name)
        print('encodeListKnown len: ', len(encodeListKnown))

        #выводим имя на фото
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 1, y2 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

        #пишем в лог
        markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

#cap.release()
#cv2.destroyAllWindows()
