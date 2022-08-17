import cv2
import dlib

rgb_image = cv2.imread("picture-this.jpg")

if rgb_image is None:
	print("Could not read input image")
	exit()
	
	
# The function for performin HOG face detection
hog_face_detector = dlib.get_frontal_face_detector()


# Set the dimensions of the image
#dims = rgb_image.shape



# Detect the faces in the image
face_rectangles = hog_face_detector(rgb_image, 1) 

# For each detected face, extract the depth from the depth image
for face_rectangle in face_rectangles:
    print('Faces were detected')

    # The coordinates of the rectanle
    x1 = face_rectangle.left()
    x2 = face_rectangle.right()
    y1 = face_rectangle.top()
    y2 = face_rectangle.bottom()

    # Extract region containing face
    #face_region = rgb_image[y1:y2,x1:x2]

    # Visualize the extracted face
    # cv2.imshow("Depth window", face_region)
    # cv2.waitKey(1)
    
    cv2.rectangle(rgb_image, (x1,y1), (x2,y2), (0,255,0), 2)
    
cv2.imshow("Image", rgb_image)
cv2.waitKey()

