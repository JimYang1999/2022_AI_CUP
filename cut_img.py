#TEAM_1970

import pandas as pd
import cv2
import os 
import time

#讀取excel，以便取得準心位子
def load_file(path):
    df = pd.read_csv(path,encoding= 'unicode_escape')
    excel_file_name = list(df.iloc[:,1])
    excel_file_class = list(df.iloc[:,-1])
    excel_file_target_x = list(df.iloc[:,2])
    excel_file_target_y = list(df.iloc[:,3])
    return excel_file_name,excel_file_class,excel_file_target_x,excel_file_target_y


def make_folder(class_):
    class_ = list(set(class_))
    path = './train/'
    for i in class_:
        os.makedirs(path+i)

#剪裁方式1：取得最小邊與準心，之後以準心的位子向外切最小邊的正方形
def cut_img(name,class_,x,y):
    #method 1 
    img = cv2.imread(f'./dataset/{class_}/{name}')
    name = name[:-4]
    row,col , chan = img.shape
    print(f"col = {col} , row = {row}")
    
    mins = int(min(row,col) / 2)
    c_x , c_y = int(row/2)+x , int(col/2)+y
    print(f'lt_x = {c_x-mins} \nlt_y = {c_y-mins}\nrd_x = {c_x+mins}\n rd_y = {c_y+mins}')
    lt_x = c_x-mins if c_x-mins>0 else 0
    lt_y = c_y-mins if c_y-mins>0 else 0
    rd_x = c_x+mins 
    rd_y = c_y+mins 
    img = img[lt_x:rd_x,lt_y:rd_y]
    img = cv2.resize(img,(720,720))
    print(img.shape[:2])
    cv2.imwrite(f'./test/{class_}/{name}-1.jpg',img)
    print(f'{class_} , {name} done')
    #method 1
 
#剪裁方式2：取得準心，並由準心的位置長寬各加500進行裁減
def cut_img2(name,class_,x,y):
    #method 2
    img = cv2.imread(f'./dataset/{class_}/{name}')
    name = name[:-4]
    row, col , chan = img.shape
    x_row = x_col = y_row = y_col = 0
    x_row=int(row*0.5+x+500) if int(row*0.5+x+500)>0 and int(row*0.5+x+500)<row  else row
    x_col=int(col*0.5+y+500) if int(col*0.5+y+500)>0 and int(col*0.5+y+500)<col else col
    y_row=int(row*0.5-500)+x if int(row*0.5-500)+x<x_row and int(row*0.5-500)+x>0 else 0
    y_col=int(col*0.5-500)+y if int(col*0.5-500)+y<x_col and int(col*0.5-500)+y>0 else 0
    print(y_row,x_row,y_col,x_col)
    img = img[y_row:x_row,y_col:x_col]
    print(name)
    img = cv2.resize(img,(640,640))
    #method 2
    cv2.imwrite(f'./test/{class_}/{name}-2.jpg',img)
    print(f'{class_} , {name} done')
    
start = time.time()
path = './dataset/training/tag_locCoor.csv'

name , class_ ,x,y = load_file(path)
num=1
folder=1
if folder:make_folder(class_)

for i in range(len(name)):
    cut_img(name[i],class_[i],x[i],y[i])
    cut_img2(name[i],class_[i],x[i],y[i])
    num+=1
end=time.time()
print(f'總共{num}張，共花{end-start}')