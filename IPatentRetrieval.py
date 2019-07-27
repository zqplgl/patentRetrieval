import cv2
import os
from objTypeClassifier import ObjTypeClassifier
import numpy as np
from annoy import AnnoyIndex

class PatentRetrieval:
    def __init__(self, model_dir, gpu_id=0):
        deploy_file = os.path.join(model_dir,"patentRetrieval/deploy.prototxt")
        assert os.path.exists(deploy_file)

        weights_file = os.path.join(model_dir,"patentRetrieval/final.caffemodel")
        assert os.path.exists(weights_file)

        self.__extracter = ObjTypeClassifier(deploy_file,weights_file,gpu_id)
        self.__feature_len = 1024

    def create_retrieval_model(self, retrieval_model_path):
        assert (os.path.exists(retrieval_model_path))
        retrieval_model = AnnoyIndex(self.__feature_len, metric="angular")
        retrieval_model.load(retrieval_model_path)

        return retrieval_model

    def extractFeature(self,im):
        try:
            feature = self.__extracter.extractFeature(im, 'loss2/fc')
        except:
            return []

        return feature

    def buildRetrievalDatabase(self,features,retrieval_model_path):
        self.__database = AnnoyIndex(self.__feature_len,metric="angular")
        index = 0
        for feature in features:
            self.__database.add_item(index,feature)
            index += 1

        self.__database.build(2*self.__feature_len)
        self.__database.save(retrieval_model_path)

    def query(self,im, k, retrieval_model):
        feature = self.extractFeature(im)
        res = retrieval_model.get_nns_by_vector(feature,k,include_distances=False)
        return  res

def buildRetrievalDatabase():
    model_dir = "./models"
    net = PatentRetrieval(model_dir)

    features = []
    index = 0
    pic_dir = "/home/zqp/testpic/patentRetrieval"
    num = len(os.listdir(pic_dir))
    for pic_name in os.listdir(pic_dir):
        im = cv2.imread(os.path.join(pic_dir,pic_name),0)
        res = net.extractFeature(im)
        if len(res):
            features.append(res)
        else:
            print (os.path.join(pic_dir,pic_name)+"************* is not exist")

        index += 1
        print ("processed***************%s/%s"%(index,num))

    net.buildRetrievalDatabase(features,"./123.tree")

def query():
    model_dir = "./models"
    retrieval_model_path="./123.tree"
    net = PatentRetrieval(model_dir)
    retrieval_model = net.create_retrieval_model(retrieval_model_path)

    pic_dir = "/home/zqp/testpic/patentRetrieval"
    for pic_name in os.listdir(pic_dir):
        print(pic_name)
        im = cv2.imread(os.path.join(pic_dir,pic_name),0)

        res = net.query(im,5,retrieval_model)
        print (res)


if __name__=="__main__":
    # buildRetrievalDatabase()
    query()

