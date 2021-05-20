# Overview of the files

## Attendance Project
1. Uses 128 face encodings per image to detect faces. These encodings are used to create bounding boxes around the detected faces. Face encodings are detected using face_recognition module.
2. A list of pre-defined images establishes the faces to be detected real time using webcam footage. Minimum one image per person to be detected is needed.
3. The datetime module is used to save the name of the person detected and the time of detection for attendance purposes in a csv file.

## Face Landmark Filtering
You can download the required pre-trained face detection model for Face Landmark Filtering [Here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

Ever had a problem deciding which lipstick shade would suit you the best? Here is a simple yet fun project can help you decide:p
### Approach
1. The basic idea is that we will come up with 68 specific points (called landmarks) that exist on every face.
2. Using these facial landmarks we will crop out different features of the face i.e eyebrows, lips, eyes, nose etc.
3. Now we focus on our cropped lips and segment them out by creating a mask. 
4. We color this mask using Trackbars wherein we can play with a wide range of colors (rgb 0-255) and overlay this colored mask on our original image. 
5. Thus what we finally get is our original image window and trackbars window, the values of which can be played with in real time. The corresponding changes will simultaneously be visible on the image!!

## HOG_Face_Detector
Multiple face detection using dlib library
