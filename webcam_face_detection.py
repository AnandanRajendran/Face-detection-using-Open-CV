import cv2
import numpy as np


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("/Users/anandanr/Docs/Code/DL/face detection/haarcascade_frontalface_default.xml")
faces_list = []
while True:
    ret,frame = cap.read()
    if not ret:
        cap.release()
        cv2.destroyAllWindows()
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    data = face_cascade.detectMultiScale(gray,minNeighbors=12)
    if len(data)==0:
        cv2.putText(frame,"No Face Detected",(100,100),cv2.FONT_HERSHEY_COMPLEX,1,color=(0,0,255))
    else:
        cv2.putText(frame,f"{len(data)} face detected",(100,100),cv2.FONT_HERSHEY_COMPLEX,2,color=(0,0,255))
        for i in data:
            x,y,w,h=i
            faces=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            face=frame[y:y+h,x:x+w]
            face_res=cv2.resize(face,(100,100))
            faces_list.append(face_res)
    cv2.imshow("ing",frame)
    if cv2.waitKey(10)==ord("k"):
        break
cv2.destroyAllWindows()
cap.release()

print(np.array(face).shape)