import cv2
import os
import numpy as np

def train_vocabulary(imgs, detector, extractor, cluters = 10):
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(cluters)
    for img in imgs:
        description = extractor.compute(img, detector.detect(img))[1]
        if description is not None:
            bow_kmeans_trainer.add(description)
    voc = bow_kmeans_trainer.cluster()
    return voc

if __name__ == '__main__':
    traindir = 'C:\workspace\python\object_detect_svm\data\CarData\TrainImages'
    testdir = 'C:\workspace\python\object_detect_svm\data\CarData\TestImages'

    print('load images...')
    pos_imgs = []
    neg_imgs = []
    for f in os.listdir(traindir):
        img = cv2.imread(os.path.join(traindir,f), cv2.IMREAD_GRAYSCALE)
        if 'pos-' in f:
            pos_imgs.append(img)
        else:
            neg_imgs.append(img)
    print('done postive:%d, negtive:%d'%(len(pos_imgs),len(neg_imgs)))

    detector = cv2.xfeatures2d.SIFT_create()
    extractor = cv2.xfeatures2d.SIFT_create()

    flann_params = dict(algorithm=1, trees=5)
    flann = cv2.FlannBasedMatcher(flann_params, {})
    extract_bow = cv2.BOWImgDescriptorExtractor(extractor, flann)

    ## train bow vocabulary
    print('train vocabulary...')
    cluters = 100 # length of bow
    voc_train_len = 10 # use some pos/neg images to train voc
    voc = train_vocabulary(pos_imgs[:voc_train_len]+neg_imgs[:voc_train_len], detector, extractor, cluters)
    extract_bow.setVocabulary(voc)

    ## compute BoW for each image
    print('compute BoW...')
    traindatas, trainlabels = [],[]
    for img in pos_imgs:
        bow = extract_bow.compute(img, detector.detect(img))
        if bow is not None:
            traindatas.extend(bow)
            trainlabels.append(1)
    for img in neg_imgs:
        bow = extract_bow.compute(img, detector.detect(img))
        if bow is not None:
            traindatas.extend(bow)
            trainlabels.append(-1)

    ## train SVM, with input: BoW output:-1,1
    print('train SVM...')
    svm = cv2.ml.SVM_create()
    svm.train(np.array(traindatas), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

    print('evaluate SVM...')
    TP,TN,FP,FN = 0,0,0,0
    for img in pos_imgs:
        bow = extract_bow.compute(img, detector.detect(img))
        p = -1 if bow is None else svm.predict(bow)[1][0][0]
        if p == -1:
            FN+=1
        else:
            TP+=1
    for img in neg_imgs:
        bow = extract_bow.compute(img, detector.detect(img))
        p = -1 if bow is None else svm.predict(bow)[1][0][0]
        if p == -1:
            TN+=1
        else:
            FP+=1
    accuracy = float(TP)/(TP+FP)
    recall = float(TP)/(TP+FN)
    print('accuracy:%.2f%% recall:%.2f%%'%(accuracy*100,recall*100))

    ## test svm
    for f in os.listdir(testdir):
        img = cv2.imread(os.path.join(testdir,f), cv2.IMREAD_GRAYSCALE)
        cv2.imshow('img',img)
        key = cv2.waitKey()
        if key==27:
            break