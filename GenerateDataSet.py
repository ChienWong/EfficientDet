import cv2, numpy as np, tensorflow as tf
from tensorflow._api.v2 import data
from tensorflow.keras.utils import to_categorical
import json
from tensorflow.python.ops.gen_array_ops import transpose
import utils
from Config import Config

class DataSetCoco():
    def __init__(self,Jsonfile,Imagedir,config=Config):
        self.NUM_CLASS = config.NUM_CLASS
        self.image_size=config.IMAGE_SIZE
        with open(Jsonfile) as f:
            js = json.load(f)
        dict = {}
        for i in js['annotations']:
            if dict.get(str(i['image_id'])) == None:
                dict[str(i['image_id'])] = [i]
            else:
                dict[str(i['image_id'])].append(i)
        
        dictImage = {}
        for i in js['images']:
            dictImage[str(i['id'])] = i

        if Imagedir[(-1)] == '/':
            Imagedir.rstrip('/')

        ImageList = []
        for i in dict.items():
            iscrowd = False
            for j in i[1]:
                if j['iscrowd'] == 1:
                    iscrowd = True
                    break
            if iscrowd == False:
                ImageList.append((Imagedir + '/' + dictImage[i[0]]['file_name'],i[1]))
        self.ImageList=ImageList

    def Generate(self,id:tf.Tensor):
        path=self.ImageList[id.numpy()][0]
        img = cv2.imread(path)
        bbox = []
        category=[]
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=(np.int8))
        for i in self.ImageList[id.numpy()][1]:
            bbox.append(i['bbox'])
            category.append(i['category_id'])
            for j in i['segmentation']:
                polygon = np.asarray(j, np.int32).reshape(1, -1, 2)
                cv2.fillPoly(mask, polygon, color=(i['category_id']))
        bbox = np.asarray(bbox, dtype=(np.float32))
        category = np.asarray(category, dtype=(np.int32))
        mask = to_categorical(mask, (self.NUM_CLASS + 1), dtype=(np.int16))
        img,bboxs=utils.transformImageSize(img,self.image_size,bbox)
        anchors =utils.anchors_for_shape(self.image_size)
        bboxs[:,2]=bboxs[:,2]+bboxs[:,0]
        bboxs[:,3]=bboxs[:,3]+bboxs[:,1]
        annotations=np.concatenate((bboxs,category[:,None]),axis=1)
        labels,regressions=utils.anchor_targets_bbox(anchors,img,annotations,self.NUM_CLASS)
        target=np.concatenate((labels,regressions),axis=1)
        return img,labels,regressions
    
    def Divide(self,img,labels,regressions):
        return img,(labels,regressions)

    def getDataSet(self):
        dataset = tf.data.Dataset.range(0,len(self.ImageList))
        dataset = dataset.map(lambda x:tf.py_function(func=self.Generate,inp=[x],Tout=(tf.int32,tf.float32,tf.float32)),
                    num_parallel_calls=(tf.data.AUTOTUNE),deterministic=False)
        dataset = dataset.map(self.Divide,num_parallel_calls=(tf.data.AUTOTUNE),deterministic=False)
        return dataset

NUM_CLASS_Cifar = 100
def GenerateDataCifar(x, y):
    label = tf.Variable(lambda : tf.zeros(NUM_CLASS_Cifar+1, tf.int32))
    label.assign(tf.zeros(NUM_CLASS_Cifar+1, tf.int32))
    label[x].assign(1)
    img = tf.reshape(y, [3, 32, 32])
    img = tf.transpose(img, [1, 2, 0])
    return (img, label)


def getDataSetCifar(file):
    dict = {}
    with open(file, 'rb') as (f):
        import pickle
        dict = pickle.load(f, encoding='bytes')
    dataset = tf.data.Dataset.from_tensor_slices((dict[b'fine_labels'], dict[b'data']))
    dataset = dataset.map(GenerateDataCifar)
    return dataset

