#TEAM_1970

from os import listdir
from os.path import isfile, isdir, join
import shutil
import os

def search_file(path): #return all file name
    file=[]
    for i in os.listdir(path):
        i = i[:-4]
        file.append(i)
    return file

def makedir(save_path,dir_):
    for i in dir_:
        if not os.path.isdir(save_path + i):
            os.mkdir(save_path + i)

def move_file(source , train , test ,dir_):
    for d in dir_:
        path = source + d
        file = search_file(path)
        for i in range(len(file)):
            if i < int(len(file)*0.9):
                shutil.copyfile(path + '/' +  file[i] + '.jpg', train + d +'/' + file[i] + '.jpg')
            else:
                shutil.copyfile(path + '/' +  file[i] + '.jpg', test + d +'/' + file[i] + '.jpg')
        
source_path = './image_cut/'
save_path_train = './cnn/train/'
save_path_test = './cnn/test/'
dir_ = listdir(source_path)
makedir(save_path_train , dir_) #建立資料夾
makedir(save_path_test , dir_)
move_file(source_path , save_path_train , save_path_test ,dir_)
