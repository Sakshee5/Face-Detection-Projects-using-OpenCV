"""
Facial Landmarks:
The basic idea is that we will come up with 68 specific points (called landmarks) that exist on every face — the top of the chin,
the outside edge of each eye, the inner edge of each eyebrow, etc. Now that we know where the eyes and mouth are, we’ll simply rotate,
scale and shear the image so that the eyes and mouth are centered as best as possible.
"""
import cv2
import numpy as np
import dlib

webcam = False

cap = cv2.VideoCapture(0)

# detector for detecting faces and for drawing corresponding bounding boxes
# predictor for getting the landmarks (68 unique values)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/Sakshee/Documents/shape_predictor_68_face_landmarks.dat")

def empty(a):
    pass

# Trackbars to play with colors of a particular feature (say lips)
cv2.namedWindow('BGR')
cv2.resizeWindow('BGR', 320, 120)
cv2.createTrackbar('Blue', 'BGR', 0, 255, empty)
cv2.createTrackbar('Green', 'BGR', 0, 255, empty)
cv2.createTrackbar('Red', 'BGR', 0, 255, empty)

def createBox(img, points, scale=5, masked = False, cropped = True):
    if masked:
        mask = np.zeros_like(img)                              # same image as original, just full black
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))   # fill the particular/required feature in white
        img = cv2.bitwise_and(img, mask)                       # gives the mask with the actual color of the lips

    if cropped:
        # to extract cropped image of a particular feature
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y:y+h, x:x+w]
        imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale )
        return imgCrop

    else:
        return mask

while True:

    if webcam:
        success, img = cap.read()
        imgOriginal = img.copy()

    else:
        img = cv2.imread('C:/Users/Sakshee/Pictures/Screenshot (170).png')
        img = cv2.resize(img, (0, 0), None, 0.32, 0.32)
        imgOriginal = img.copy()


    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)
    # print(faces)
    # prints rectangles[[(302, 267) (611, 577)]] = [(face.left(), face.top()), (face.right(), face.bottom())]


    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        # to draw the box around the detected face
        # imgOriginal = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(imgGray, face)

        # contains a list of co-ordinates of 68 landmarks for all faces detected
        myPoints = []

        # since we get 68 landmarks per face
        # landmarks.part(n) gives the co-ordinate of the nth landmark
        # landmarks.part(n).x gives the x co-ordinate of the nth landmark
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            myPoints.append([x,y])

            #cv2.circle(imgOriginal, (x,y), 3, (50, 50, 255), cv2.FILLED)
            #cv2.putText(imgOriginal, str(n), (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 255), 1)

        # type list gives an error when accessing elements [a:b] thus conversion to array necessary
        myPoints = np.array(myPoints)

        # imgEyeBrowLeft = createBox(img, myPoints[17:22])
        # imgEyeBrowRight = createBox(img, myPoints[22:27])
        # imgNose = createBox(img, myPoints[27:36])
        # imgLeftEye = createBox(img, myPoints[36:42])
        # imgRightEye = createBox(img, myPoints[42:48])
        imgLips = createBox(img, myPoints[48:61], masked=True, cropped=False)

        imgColorLips = np.zeros_like(imgLips)
        b = cv2.getTrackbarPos('Blue', 'BGR')
        g = cv2.getTrackbarPos('Green', 'BGR')
        r = cv2.getTrackbarPos('Red', 'BGR')
        imgColorLips[:] = b, g, r
        # imgColorLips[:] = 153, 0, 157                       # creates an image same size of imgLips of purple background
        imgColorLips = cv2.bitwise_and(imgLips, imgColorLips) # adds the purple color in the masked region of lips

        # adding blur to give it a more natural look
        imgColorLips = cv2. GaussianBlur(imgColorLips, (7, 7), 10)
        imgOriginalGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)     # to see the lip color against black-white contrast

        # since the final image has purple color we need 3 channels
        imgOriginalGray = cv2.cvtColor(imgOriginalGray, cv2.COLOR_GRAY2BGR)

        imgColorLips = cv2.addWeighted(imgOriginal, 1, imgColorLips, 0.3, 0)


        # cv2.imshow('Left Eyebrow', imgEyeBrowLeft)
        # cv2.imshow('Right Eyebrow', imgEyeBrowRight)
        # cv2.imshow('Nose', imgNose)
        # cv2.imshow('Left Eye', imgLeftEye)
        # cv2.imshow('Right Eye', imgRightEye)
        cv2.imshow('Lips', imgLips)
        cv2.imshow('BGR', imgColorLips)

        # print(myPoints)


    cv2.imshow('Original', imgOriginal)
    cv2.waitKey(1)