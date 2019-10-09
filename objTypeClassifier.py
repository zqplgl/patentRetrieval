#coding=utf-8
import caffe
import numpy as np
import cv2
import os
import pickle
import shutil

class ObjTypeClassifier:
    def __init__(self,prototxt,weightfile,gpu_id=0):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        self.__net = caffe.Net(prototxt,weightfile,caffe.TEST)

        inputBlobShape =  self.__net.blobs['data'].data[0].shape
        self.__input_geometry = (inputBlobShape[-1],inputBlobShape[-2])

    def __getInputBlob(self,im):
        im_resize = cv2.resize(im,self.__input_geometry).astype(np.float32)/256
        blob = np.zeros((1,1,self.__input_geometry[1],self.__input_geometry[0]),dtype=np.float32)
        blob[0,:,:,:] = im_resize
        blob = blob.astype(np.float32)

        return blob

    def __cleanBounding(self,im):
        ret, im_threshold = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(im_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        con = np.vstack(contours)
        con = cv2.convexHull(con)
        con = np.squeeze(np.array(con), axis=(1,))
        con = cv2.boundingRect(con)

        return im[con[1]:con[1]+con[3],con[0]:con[0]+con[2]]

    def __resize_data(self,im):
        im = self.__cleanBounding(im)
        # cv2.imshow("im",im)
        # cv2.waitKey(0)
        h, w = im.shape

        max_side = max(w, h)
        im_resize = (np.ones((max_side, max_side)) * 255).astype(np.uint8)
        if h < w:
            up = int((w - h) / 2 + 0.7)
            im_resize[up:up + h] = im

        elif w < h:
            left = int((h - w) / 2 + 0.7)
            im_resize[:, left:left + w] = im
        else:
            im_resize = im

        return im_resize

    def extractFeature(self,im,featureBlobName):
        im_resize = self.__resize_data(im)
        self.__net.blobs['data'].data[...] = self.__getInputBlob(im_resize)
        self.__net.forward()
        feature = self.__net.blobs[featureBlobName].data.flatten()

        return feature

    def classify(self,im,featureBlobName):
        self.__net.blobs['data'].data[...] = self.__getInputBlob(im)
        self.__net.forward()
        prob = self.__net.blobs['prob'].data.flatten()
        index = prob.argsort()[-1]
        confidence = prob[index]
        feature = self.__net.blobs[featureBlobName].data.flatten()

        return (index,confidence,feature)

def run():
    cv2.namedWindow("base",0)
    cv2.namedWindow("im",0)

    classifier = initNet()

    pic_path = [line.strip() for line in open("./pic_path.txt").readlines()]
    base_dir = "/media/zqp/data/train_data/patent_retrieval/"
    base_path = [base_dir+"%06d/%06d.jpg"%(index,index) for index in range(1000)]
    for path in base_path:
        im = cv2.imread(path)
        result = classifier.classify(im,'loss2/fc')
        print (result[0],"****************",result[1])
        cv2.imshow("base",cv2.imread(base_path[result[0]]))
        cv2.imshow("im", im)
        if cv2.waitKey(0)==27:
            break

def initNet():
    prototxt = r'./deploy.prototxt'
    weightfile = r'./final.caffemodel'
    classifier = ObjTypeClassifier(prototxt,weightfile,0)

    return classifier

def run_1():
    cv2.namedWindow("im",0)
    classifier = initNet()
    pic_path = [line.strip() for line in open("./pic_path.txt").readlines()]
    for path in pic_path:
        print(path)
        im = cv2.imread(path, 0)
        result = classifier.classify(im,'loss2/fc')
        print (result[0],"****************",result[1])
        cv2.imshow("im",im)
        if cv2.waitKey(0)==27:
            break

def extraceBaseFeatures():
    features = []
    classifier = initNet()
    pic_path = [line.strip() for line in open("./pic_path.txt").readlines()]
    index = 0
    for path in pic_path:
        im = cv2.imread(path, 0)
        feature = classifier.extractFeature(im,'loss2/fc')
        features.append(feature)
        index+=1

        if index%10==0:
            print ("process img********"+str(index))

    features = np.array(features)
    array1 = np.sqrt(np.sum(features*features,1))
    features = (features.T/array1).T

    file1 = open("./features1.pkl","wb")
    pickle.dump(features, file1)
    fs = cv2.FileStorage("./features1.json.gz",cv2.FILE_STORAGE_WRITE)
    fs.write("mat", features)
    fs.release()

def classify():
    features = pickle.load(open("./features.pkl","rb"))
    base_path = [line.strip() for line in open("./features.txt").readlines()]

    classifier = initNet()
    pic_path = [line.strip() for line in open("./pic_path.txt").readlines()]

    cv2.namedWindow("im", 0)
    cv2.namedWindow("base", 0)
    for path in pic_path:
        print(path)
        im = cv2.imread(path, 0)
        feature = classifier.extractFeature(im,'loss2/fc')

        feature = feature/np.sqrt(np.dot(feature,feature))
        result = np.matmul(features, feature)
        indexs = np.flip(np.argsort(result))

        # indexs = np.where(result>0.9)[0]

        if len(indexs)<1:
            continue

        flag = False
        cv2.imshow("im", im)
        i = 1
        for index in indexs[:10]:
            cv2.imshow("base", cv2.imread(base_path[index]))
            print("top*******%s/%s"%(result[index], i))
            i += 1

            if cv2.waitKey((0))==27:
                flag = True
                break

        if flag:
            break

def classify_1():
    features = pickle.load(open("./features.pkl","rb"))
    base_path = [line.strip() for line in open("./features.txt").readlines()]
    classifier = initNet()
    pic_save_root_dir = "/media/zqp/data/train_data/patent_retrieval/"
    pic_path = [line.strip() for line in open("./pic_path.txt").readlines()]
    pic_index = 0
    available = 0

    count = 0
    for path in pic_path:
        im = cv2.imread(path)
        feature = classifier.extractFeature(im,'loss2/fc')

        feature = feature/np.sqrt(np.dot(feature,feature))
        result = np.matmul(features, feature)
        indexs = np.argsort(result)

        count += 1
        if count%10==0:
            print("process************%s×××××××available: %s"%(count,available))
        if result[indexs[-1]]<0.85:
            continue

        label = os.path.basename(os.path.dirname(base_path[indexs[-1]]))
        shutil.move(path, pic_save_root_dir+label)
        available += 1

if __name__ == '__main__':
    extraceBaseFeatures()

