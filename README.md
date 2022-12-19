# 農地作物現況調查影像辨識競賽 – 秋季賽： AI作物影像判釋
## TEAM_1970
---
### 資料前處理 read_csv.py

讀取官方給予的tag_loccoor_private.csv，將所有檔案名稱記錄起來後，進入資料夾內搜尋，看該檔案名稱所屬什麼類別，會寫到新建的excel內，output會像以下表格：

|  filename   | class  |
|  :----:  | :----:  |
| 123.jpg  |0 |
| 456.jpg  | 1 |

或

|  filename   | class  |                  
|  :----:  | :----:  |
| 123.jpg  | asparagus |
| 456.jpg  | betel |

得到此excel後，可以將內容複製起，貼至tag_loccoor_private.csv，這樣即可知道該檔案名稱屬於何種類別。

原先的tag_loccor_private.csv:

|  TARGET   | Img  | target_x | target_y | COUNTYNAME |　TOWNNAME | town_x | town_y |
|  :----:  | :----:  |  :----:  | :----:  |  :----:  | :----:  |  :----:  | :----:  | 
| 0  |0000f284-d8fa-4f88-b1f2-ef7769e285d5.jpg | 0 | 0 | 彰化縣 | 溪湖鎮 | 120.4831848 | 23.9518013 |
| 1  |0006ecdf-82ed-46b8-86b1-f4e94ae83508.jpg | 0 | -144 | 彰化縣 | 埔心鄉 | 120.5342712 | 23.9528713 |

新增class後，會變成:

|  TARGET   | Img  | target_x | target_y | COUNTYNAME |　TOWNNAME | town_x | town_y | class |
|  :----:  | :----:  |  :----:  | :----:  |  :----:  | :----:  |  :----:  | :----:  | :----:  | 
| 0  |0000f284-d8fa-4f88-b1f2-ef7769e285d5.jpg | 0 | 0 | 彰化縣 | 溪湖鎮 | 120.4831848 | 23.9518013 | 0 |
| 1  |0006ecdf-82ed-46b8-86b1-f4e94ae83508.jpg | 0 | -144 | 彰化縣 | 埔心鄉 | 120.5342712 | 23.9528713 | 0 |


### 資料前處理 cut_img.py

我們會先讀取前項新增完class後的tag_loccor_private.csv，並透過load_file函數輸出 1. 檔案名稱, 2. 類別, 3. x座標, 4.y座標 

有了前四項資料，即可裁減圖片，每張圖片會有兩種剪裁方式。

### 資料前處理 split_file.py

前項圖片剪裁完畢後，我們可以使用split_file.py將檔案分成訓練集及測試集

source_path為前項圖片剪裁完後的資料夾位置，之後會透過程式將前9成的檔案分配至./train資料夾，其餘的分至./test資料夾。

### 資料前處理 cut_for_others.py

此程式專門處理others的圖片，會將others每張圖片切割為1. 右上角 2. 左下角 3. 原圖以最小邊切為正方形。

---

### 訓練流程 train_cnn.py

程式內需要設置的地方有
1. train_dir 此為訓練資料集的資料夾位置
2. test_dir 此為測試資料集的資料夾位置
3. 其餘皆為超參數，程式碼為最終超參數的設置。
4. 若要繼續訓練，可將第31行註解解除，並將30行註解起來。
5. 模型每20次及最大的test_acc皆會儲存。

---

### 預測 predict_testing.py

程式內需要設置的地方有
1. model_path 此為model的位置
2. data_dir 要測試的圖片資料夾位置
3. 64行的部份我們先自己建立一個空的xlsx，再將預測結果放至此xlsx內，結束後再將結果複製貼上至主辦方提供的csv內。


