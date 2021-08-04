import tensorflow as tf
import tensorflow.keras as keras
import EfficientNet
import GenerateDataSet
import numpy as np
import utils
import EfficientDet
from EfficientDet import EfficientDet
from Loss import smooth_l1, focal
from tensorflow.keras.utils import to_categorical
from Config import Config

callbacks=[
        keras.callbacks.TensorBoard(log_dir="log",histogram_freq=0, write_graph=True, write_images=False),
        keras.callbacks.ModelCheckpoint("log"+"/EfficentDet{epoch:02d}.h5",verbose=0, save_weights_only=True)]

model=EfficientDet(Config.phi,num_classes=Config.NUM_CLASS)
trainDataset=GenerateDataSet.DataSetCoco(
    "/home/wang/Data/coco/annotations/instances_train2017.json","/home/wang/Data/coco/train2017").getDataSet()
valDataset=GenerateDataSet.DataSetCoco(
    "/home/wang/Data/coco/annotations/instances_val2017.json","/home/wang/Data/coco/val2017").getDataSet()
trainDataset=trainDataset.cache().batch(1)
valDataset=valDataset.cache().batch(1)

for i  in range(1,[227, 329, 329, 374, 464, 566, 656][Config.phi]):
    model.layers[i].trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=[focal(),smooth_l1()])
model.fit(x=trainDataset,epochs=100,validation_data=valDataset,callbacks=callbacks,shuffle=True)

model.summary()

# model=EfficientNet.EfficientNetB6(classes=101,weights=None)
# datasetTrain=GenerateDataSet.getDataSetCifar("/home/wang/Data/cifar/cifar-100-python/train").batch(32)
# datasetVal=GenerateDataSet.getDataSetCifar("/home/wang/Data/cifar/cifar-100-python/test").batch(32)
# model.summary()
# from tensorflow.keras.utils import plot_model
# plot_model(model,to_file="model.png",show_shapes=True)
# x=np.arange(1536*1536*3).reshape(1,1536,1536,3)
# y=model.predict(x)
# print(y)
# print(np.sum(y))
# train,val=tf.keras.datasets.cifar100.load_data()
# model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(x=datasetTrain,epochs=100,validation_data=datasetVal,callbacks=callbacks,shuffle=True)