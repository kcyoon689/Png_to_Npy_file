import numpy as np
from PIL import Image
import os


currentDirPath = os.getcwd()
rawDataDirPath = currentDirPath + '/train/png_label'
rawDataFilePath_list = os.listdir(rawDataDirPath)
rawDataFileFullPath_list = [rawDataDirPath + '/' + file_name for file_name in rawDataFilePath_list]
# saveDataDirPath=os.getcwd() + "/train_img"

# print(len(rawDataFileFullPath_list))

for png in range(len(rawDataFileFullPath_list)):
    image = Image.open(rawDataFileFullPath_list[png])
    pixel = np.array(image, dtype='float64')
    rawDataFileFullPath_list[png] = rawDataFileFullPath_list[png].split('.')[0]
    np.save(rawDataFileFullPath_list[png] +'.npy', pixel)