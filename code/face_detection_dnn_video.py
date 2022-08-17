import cv2
import numpy as np
import sys
import re

centerx = 0
centery = 0

def click_event(event, x, y, flags, params): 
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN:
        global centerx
        global centery
        centerx = x
        centery = y



face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

vid = 'dnevna60cut.mp4'
if (len(sys.argv) > 1):
    vid = sys.argv[1]
cap = cv2.VideoCapture('../../Videos/'+vid)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('frames, distance, angle, TP , FN , FP')
frame_count = 0
dist = 4
TP,FN,FP = 0,0,0
while True:
    # Read the frame
    _, img = cap.read()
    if img is None:
        break
    dims = img.shape
    h = dims[0]
    w = dims[1]
    
    
    #user selects the starting position of the face
    if frame_count == 0:
        cv2.imshow('find_face', img)
        cv2.setMouseCallback('find_face', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    frame_count += 1
    # Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    face_detections = face_net.forward()
    
    
    face_found = 0
    face_count = 0
    # Draw the rectangle around each face
    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]
        if confidence>0.15:
            face_count += 1
            box = face_detections[0,0,i,3:7] * np.array([w,h,w,h])
            box = box.astype('int')
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            
            rec_centerx = (x1+x2)/2
            rec_centery = (y1+y2)/2
            #print(  centerx - rec_centerx )
            if abs(centerx-rec_centerx)<100 and abs(centery-rec_centery)<100:
                centerx = rec_centerx
                centery = rec_centery
                face_found += 1
    
    # Display
    cv2.imshow('img', img)
    
    
    if face_found == 0:
        FN += 1
    elif face_found == 1:
        TP += 1
    else:
        print("multiple face detections in this frame")
        
    
    FP += face_count-face_found
    
    if frame_count % (frames//4) == 0:
        #print('frames', (frames//4))
        #print('TP:',TP,', FN:',FN,', FP:',FP)
        angle = re.findall(r'\d+', vid[:-1])[0]
        print((frames//4),',', dist,',',angle, end=" , ")
        print(TP,',',FN,',',FP)
        dist -= 1
        TP,FN,FP = 0,0,0
    
    
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()
