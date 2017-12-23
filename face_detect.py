import cv2

img = cv2.imread('data/1.jpg',cv2.IMREAD_GRAYSCALE)
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(img)
show_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for x,y,w,h in faces:
    show_img = cv2.rectangle(show_img, (x,y), (x+w,y+h), (255,0,0), 2)
cv2.imshow('img',show_img)
cv2.waitKey()