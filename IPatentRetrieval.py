import cv2
import os
from objTypeClassifier import ObjTypeClassifier
import faiss
import numpy as np

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

        retrieval_model = faiss.read_index(retrieval_model_path)
        return retrieval_model

    def extractFeature(self,im):
        try:
            feature = self.__extracter.extractFeature(im, 'loss2/fc')
        except:
            return []

        return feature

    def buildRetrievalDatabase(self,features,retrieval_model_path):
        if os.path.exists(retrieval_model_path):
            retrieval_model = faiss.read_index(retrieval_model_path)
        else:
            retrieval_model = faiss.IndexFlatL2(self.__feature_len)


        features = np.array(features)
        retrieval_model.add(features)
        faiss.write_index(retrieval_model,retrieval_model_path)

        return retrieval_model

    def query(self,im, k, retrieval_model):
        feature = self.extractFeature(im)

        feature = np.array([feature])
        res = retrieval_model.search(feature,k)
        return  res[1][0],res[0][0]

def buildRetrievalDatabase():
    model_dir = "./models"
    net = PatentRetrieval(model_dir)

    features = []
    index = 0
    pic_dir = "/home/zqp/testpic/patentRetrieval/"

    pic_paths = sorted([pic_dir+pic_name for pic_name in os.listdir(pic_dir) if pic_name.endswith(".jpg")])
    num = len(os.listdir(pic_dir))
    for pic_path in pic_paths:
        im = cv2.imread(pic_path,0)
        res = net.extractFeature(im)
        if len(res):
            features.append(res)

        index += 1
        print ("processed***************%s/%s"%(index,num))

    net.buildRetrievalDatabase(features,"./123.index")

def query():
    model_dir = "./models"
    retrieval_model_path="./123.index"
    net = PatentRetrieval(model_dir)
    retrieval_model = net.create_retrieval_model(retrieval_model_path)

    pic_dir = "/home/zqp/testpic/patentRetrieval/"
    pic_paths = sorted([pic_dir+pic_name for pic_name in os.listdir(pic_dir) if pic_name.endswith(".jpg")])
    for pic_name in os.listdir(pic_dir):
        print(pic_name)
        im = cv2.imread(os.path.join(pic_dir,pic_name),0)

        cv2.imshow("base",im)

        res = net.query(im,5,retrieval_model)

        top = 0
        for idx in res[0]:
            im_query = cv2.imread(pic_paths[idx])
            cv2.imshow("query",im_query)
            top += 1

            print ("top*********",top)
            cv2.waitKey(0)




if __name__=="__main__":
    # buildRetrievalDatabase()
    query()

