import cv2
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


# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
#cap = cv2.VideoCapture(0)
# To use a video file as input
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    face_found = 0
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        rec_centerx = x+w/2
        rec_centery = y+h/2
        #print(  centerx - rec_centerx )
        if abs(centerx-rec_centerx)<100 and abs(centery-rec_centery)<100:
            centerx = rec_centerx
            centery = rec_centery
            face_found += 1
            
    # Display
    cv2.imshow('img', img)
    
    
    if face_found == 0:
        FN += 1
    elif face_found >= 1:
        TP += 1
    else:
        print("negative face detections in this frame")
        
    
    FP += len(faces)-face_found
    
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
