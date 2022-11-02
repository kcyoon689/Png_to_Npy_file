import numpy as np
from PIL import Image
import os


currentDirPath = os.getcwd()
rawDataDirPath = currentDirPath + '/PngtoNpy/mask/train'
rawDataFilePath_list = os.listdir(rawDataDirPath)
rawDataFileFullPath_list = [rawDataDirPath + '/' + file_name for file_name in rawDataFilePath_list]

# print(rawDataFileFullPath_list)
# print(len(rawDataFileFullPath_list)) #train: 19980, test: 560, valid: 560

# Img = np.load(file=saveDataDirPath, allow_pickle=True)

# print(np.shape(Img))

Img=[0] * 19980

for i in range(len(rawDataFileFullPath_list)):
    Img[i] = np.load(file=rawDataFileFullPath_list[i], allow_pickle=True)

stackImg = np.stack((Img), axis=0)

print(np.shape(stackImg))
print(stackImg.dtype)

np.save("train_mask.npy", stackImg)
test = np.load(file=currentDirPath+"/train_mask.npy", allow_pickle=True)

print(np.shape(test))
print(test.dtype)