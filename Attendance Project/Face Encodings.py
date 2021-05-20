"""
Face encodings are nothing but 128 measurements for each face
For example, we might measure the size of each ear, the spacing between the eyes, the length of the nose, etc.
"""

import cv2
import face_recognition

imgElon = face_recognition.load_image_file('C:/Users/Sakshee/Pictures/Elon-Musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('C:/Users/Sakshee/Pictures/Elon Test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# without [0] it returns [(168, 425, 297, 296)]
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]

# print(faceLoc)
# prints 4 values which are basically top, right, bottom and left- (168, 425, 297, 296)
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# we are using linear svm at the backend for comparison
# results is a boolean output
results = face_recognition.compare_faces([encodeElon], encodeTest)

# if we wanna know the similarities between 2 given faces: (similar)0<value<1(un-similar)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results, faceDis)

cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)