import cv2
import os
import numpy as np

train_size = (200,200)

def face_collect(datadir,savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    save_idx = 0
    for f in os.listdir(datadir):
        img = cv2.imread(os.path.join(datadir,f), cv2.IMREAD_GRAYSCALE)
        r,c = img.shape[:2]
        if max(r,c)>500:
            img = cv2.resize(img, None, fx=500.0/max(r,c), fy = 500.0/max(r,c))
        faces = face_cascade.detectMultiScale(img)
        print(faces)
        show_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for x,y,w,h in faces:
            show_img = cv2.rectangle(show_img, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.imshow('img',show_img)
        key = cv2.waitKey()
        if key == 27:
            break
        elif key != ord('q'):
            for x,y,w,h in faces:
                savefile = os.path.join(savedir,'%d.jpg'%save_idx)
                cv2.imwrite(savefile, cv2.resize(img[y:y+h,x:x+w],train_size))
                print('save %s'%savefile)
                save_idx+=1
    print('done face collection, save %d faces'%save_idx)

def face_load(facedir):
    print('load face...')
    imgs,lbls,names = [],[],{}
    face_dir_each = sorted(os.listdir(facedir))
    for idx, name in enumerate(face_dir_each):
        names[idx] = name
        for f in os.listdir(os.path.join(facedir,name)):
            if '.pgm' in f or '.jpg' in f or '.png' in f:
                img = cv2.resize(cv2.imread(os.path.join(facedir,name,f), cv2.IMREAD_GRAYSCALE), train_size)
                imgs.append(img)
                lbls.append(idx)
    return imgs,lbls,names

def face_train(imgs,lbls,save_model = None):
    # train
    print('train...')
    model = cv2.face.EigenFaceRecognizer_create()
    # model = cv2.face.createFisherFaceRecognizer()
    model.train(np.array(imgs),np.array(lbls))
    if save_model:
        model.save(save_model)
    return model

def face_recog(img, detector, recognizer):
    faces = detector.detectMultiScale(img)
    res = []
    for x,y,w,h in faces:
        roi = cv2.resize(img[y:y+h,x:x+w], train_size)
        label,confidence = recognizer.predict(roi)
        res.append([label, confidence, x,y,w,h])
    return res

if __name__ == '__main__':
    raw_data_dir = 'C:\workspace\python\object_detect_svm\data/raw'
    face_dir = 'C:\workspace\python\object_detect_svm\data/faces'
    save_model = 'C:/workspace/python/object_detect_svm/face_model.xml'
    ## 1. collect raw images, extrace faces to train
    # face_collect(raw_data_dir,face_dir)

    ## 2. load faces and train recognize model
    imgs,lbls,names = face_load(face_dir)
    face_model = face_train(imgs,lbls)
    ## 3. test
    face_detector = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

    for f in os.listdir(raw_data_dir):
        img = cv2.imread(os.path.join(raw_data_dir,f),cv2.IMREAD_GRAYSCALE)
        r,c = img.shape[:2]
        if max(r,c)>500:
            img = cv2.resize(img, None, fx=500.0/max(r,c), fy = 500.0/max(r,c))
        res = face_recog(img, face_detector, face_model)
        show_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for label, confidence, x,y,w,h in res:
            if confidence<500:
                show_img = cv2.rectangle(show_img, (x,y), (x+w,y+h), (255,0,0), 2)
                show_img = cv2.putText(show_img, '%s:%.2f'%(names[label],confidence), (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
        cv2.imshow('res',show_img)
        key = cv2.waitKey()
        if key==27:
            break
