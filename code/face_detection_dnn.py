import cv2
import sys
import numpy as np

rgb_image = cv2.imread("picture-this.jpg")

if rgb_image is None:
	print("Could not read input image")
	exit()
	
	
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')


dims = rgb_image.shape
h = dims[0]
w = dims[1]

# Tranform image to gayscale
#gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# Do histogram equlization
#img = cv2.equalizeHist(gray)

# Detect the faces in the image
#face_rectangles = self.face_detector(rgb_image, 0)
blob = cv2.dnn.blobFromImage(cv2.resize(rgb_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
face_net.setInput(blob)
face_detections = face_net.forward()


for i in range(0, face_detections.shape[2]):
    confidence = face_detections[0, 0, i, 2]
    if confidence>0.15:
        box = face_detections[0,0,i,3:7] * np.array([w,h,w,h])
        box = box.astype('int')
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

        # Extract region containing face
        #face_region = rgb_image[y1:y2, x1:x2]

        cv2.rectangle(rgb_image, (x1,y1), (x2,y2), (0,255,0), 2)
    
cv2.imshow("Image", rgb_image)
cv2.waitKey()




