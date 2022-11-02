import numpy as np
from PIL import Image
import os


currentDirPath = os.getcwd()
rawDataDirPath = currentDirPath + '/PngtoNpy/mask/train'
rawDataFilePath_list = os.listdir(rawDataDirPath)
rawDataFileFullPath_list = [rawDataDirPath + '/' + file_name for file_name in rawDataFilePath_list]
saveDataDirPath=os.getcwd() + "/train_mask"

# print(rawDataFileFullPath_list)
# print(len(rawDataFileFullPath_list)) #train: 19980, test: 560, valid: 560

Img=[0] * 19980

for i in range(len(rawDataFileFullPath_list)):
    Img[i] = np.load(file=rawDataFileFullPath_list[i], allow_pickle=True)

stackImg = np.stack((Img), axis=0)

np.save(saveDataDirPath, "train")
# print(np.shape(stackImg))