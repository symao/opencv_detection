#coding=utf-8
import numpy as np
import cv2
import dlib
import math

def demo_cv():
    img = cv2.imread('data/1.jpg',cv2.IMREAD_GRAYSCALE)
    face_cascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img)
    show_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x,y,w,h in faces:
        show_img = cv2.rectangle(show_img, (x,y), (x+w,y+h), (255,0,0), 2)
    cv2.imshow('img',show_img)
    cv2.waitKey()

def demo_dlib():
    img = cv2.imread('data/1.jpg',cv2.IMREAD_GRAYSCALE)
    predictor_path = "./model/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)  
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)

    show_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if len(dets)>0:  
        for d in dets:
            x,y,w,h = d.left(),d.top(), d.right()-d.left(), d.bottom()-d.top()
            cv2.rectangle(show_img,(x,y),(x+w,y+h),(255,0,0),2,8,0)
            shape = predictor(img, d)
            for point in shape.parts():
                cv2.circle(show_img,(point.x,point.y), 1, color=(0,255,0))
    cv2.imshow("image",show_img)
    cv2.waitKey()

def demo_christmas():
    img = cv2.imread('C:/photo/IMG_8982.jpg')
    r,c = img.shape[:2]
    if max(r,c)>600:
        img = cv2.resize(img, None, fx=600.0/max(r,c), fy = 600.0/max(r,c))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    predictor_path = "./model/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)  
    detector = dlib.get_frontal_face_detector()
    dets = detector(gray, 1)

    hat_img = cv2.imread('C:/Users/symao/Downloads/christmas_hat.jpg')
    _,_,har_r = cv2.split(hat_img)
    _,hat_mask = cv2.threshold(har_r,10,255,cv2.THRESH_BINARY)

    def rotate(image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    if len(dets)>0:  
        for d in dets:
            x,y,w,h = d.left(),d.top(), d.right()-d.left(), d.bottom()-d.top()
            shape = predictor(gray, d)

            nose_top = np.array([shape.parts()[27].x,shape.parts()[27].y])
            nose_down = np.array([shape.parts()[30].x,shape.parts()[30].y])
            eye_left = np.array([shape.parts()[36].x,shape.parts()[36].y])
            eye_right = np.array([shape.parts()[45].x,shape.parts()[45].y])
            dir_lr = eye_right-eye_left
            dir_ud = np.array([-dir_lr[1], dir_lr[0]])
            eye_dist = np.linalg.norm(dir_lr)
            angle = math.atan2(eye_left[1]-eye_right[1], eye_left[0]-eye_right[0])+3.1415926

            hat_factor = eye_dist/hat_img.shape[0]*1.8
            small_hat_img = rotate(cv2.resize(hat_img, None, fx=hat_factor, fy=hat_factor), angle)
            small_hat_mask = rotate(cv2.resize(hat_mask, None, fx=hat_factor, fy=hat_factor), angle)

            sr,sc = small_hat_mask.shape[:2]
            ir,ic = img.shape[:2]

            add_center = nose_top-dir_ud*1.2

            for i in range(sr):
                for j in range(sc):
                    x = int(add_center[0]-sc/2+j)
                    y = int(add_center[1]-sr/2+i)
                    if small_hat_mask[i,j]>0 and x>=0 and y>=0 and x<ic and y<ir:
                        img[y,x] = small_hat_img[i,j]
    cv2.imshow("image",img)
    cv2.waitKey()


if __name__ == '__main__':
    # demo_cv()
    # demo_dlib()
    demo_christmas()