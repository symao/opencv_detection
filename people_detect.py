# coding=utf-8
import numpy as np
import cv2

img = cv2.imread('C:/data/b.jpg')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
bboxes, weights = hog.detectMultiScale(img)

for bb,weight in zip(bboxes,weights):
	x,y,w,h = bb
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	cv2.putText(img,'%.3f'%weight,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)

cv2.imshow('img',img)
cv2.waitKey()