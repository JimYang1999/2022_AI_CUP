#TEAM_1970

import os
import cv2
def search_file(path): #return all file name
    file=[]
    for i in os.listdir(path):
        i = i[:-4]
        if i not in file:
            file.append(i)
    return file
source = './dataset/others/'
source_file = search_file(source)
for i in range(int(len(source_file)*0.9)):
    img = cv2.imread(f'{source}{source_file[i]}.jpg')
    print(source_file[i])
    y, x , chan = img.shape
    print(x,y)
    center_x = x//2
    center_y = y//2
    min_=min(center_y,center_x)
    print(i)
    print(center_x,center_y)
    lt_x = center_x-min_ if center_x-min_>0 else 0
    lt_y = center_y-min_ if center_y-min_>0 else 0
    rd_x = center_x+min_ #if c_x+mins<row and c_x+mins >0 else row
    rd_y = center_y+min_ #if c_y+mins<col and c_y+mins >0 else col
    img1 = img[:min_-50,x-min_-50:x] #右上角
    img2 = img[y-min_-50:y,:min_-50] #左下角
    img = img[lt_y:rd_y,lt_x:rd_x] #原圖以最小邊切正方形
    img1 = cv2.resize(img1,(640,640))
    img2 = cv2.resize(img2,(640,640))
    img = cv2.resize(img,(640,640))
    cv2.imwrite(f'./others/{source_file[i]}-1.jpg',img1)
    cv2.imwrite(f'./others/{source_file[i]}-2.jpg',img2)
    cv2.imwrite(f'./others/{source_file[i]}.jpg',img)
