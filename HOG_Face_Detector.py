"""
HOG: Histogram of oriented gradients is method for face detection.
Intuition-
1. Replace every pixel by drawing an arrow in the direction the image is getting darker i.e arrow from that pixel to the darkest surrounding pixel
2. These arrows are called gradients and they show the flow from light to dark across the entire image
3. But this way we will get lots of arrows(gradients) so instead we’ll break up the image into small squares of say 16x16 pixels each.
4. In each square, we’ll count up how many gradients point in each major direction (how many point up, point up-right, point right, etc…).
5. Then we’ll replace that square in the image with the arrow directions that were the strongest.
"""
import dlib
from skimage import io

# upload any image with single/multiple faces
file_name = "C:/Users/Sakshee/Pictures/Image.jpg"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()

win = dlib.image_window()

# Load the image into an array
image = io.imread(file_name)

# Run the HOG face detector on the image data. The result will be the bounding boxes of the faces in our image.
detected_faces = face_detector(image, 1)

print("I found {} faces in the file {}".format(len(detected_faces), file_name))

# Open a window on the desktop showing the image
win.set_image(image)

for i, face_rect in enumerate(detected_faces):
    # Detected faces are returned as an object with the coordinates of the top, left, right and bottom edges
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

    # Draw a box around each face we found
    win.add_overlay(face_rect)

# Wait until the user hits <enter> to close the window
dlib.hit_enter_to_continue()