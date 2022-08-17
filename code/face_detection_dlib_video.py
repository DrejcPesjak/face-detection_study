import cv2
import dlib
import sys
import re

centerx = 0
centery = 0

def click_event(event, x, y, flags, params): 
    # https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN:
        global centerx
        global centery
        centerx = x
        centery = y


# The function for performin HOG face detection
hog_face_detector = dlib.get_frontal_face_detector()

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
    size = img.shape
    
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
    face_rec = hog_face_detector(img, 0)
    #print(len(face_rec))
    face_found = 0 #after loop it should be 1
    # Draw the rectangle around each face
    for fr in face_rec:
        cv2.rectangle(img, (fr.left(), fr.top()), (fr.right(), fr.bottom()), (255, 0, 0), 2)
        rec_centerx = (fr.left()+fr.right())/2
        rec_centery = (fr.top()+fr.bottom())/2
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
        
    
    FP += len(face_rec)-face_found
    
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
