import cv2
import numpy as np
from sklearn.cluster import k_means

 #二值化
def get_txt_pixels(img):
    shape = img.shape
    gray_img = to_gray(img)
    gray_img = gray_img.reshape(gray_img.size, 1)
    kmeans = k_means(gray_img, n_clusters=2)
    kmeans = kmeans[1].reshape((shape[0], shape[1]))
    zeros_pos = np.where(kmeans == 0)
    ones_pos = np.where(kmeans == 1)
    if not zeros_pos[0].size > ones_pos[0].size:
        kmeans[zeros_pos] = 255
        kmeans[ones_pos] = 0
    cv2.imwrite('c' + str(1) + '.jpg', kmeans*255)

    return kmeans*255
#灰度处理
def to_gray(img):
    """
    转变成灰度图片
    :param img:
    :return:
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#获得图片
# def get_image(root,path):
#     return  root,path

#行分割
def  get_horizontal_seg_array(txt_pixels,len_x):
    start_i = -1
    end_i = -1
    rowPairs = []
    for i in range(len_x):
        if( txt_pixels[i].any() and start_i < 0):
            start_i = i
        elif( txt_pixels[i].any()):
            end_i = i
        elif ( not txt_pixels[i].any() and start_i >= 0):
            if (end_i - start_i >= min_val):
                rowPairs.append((start_i, end_i))
                start_i, end_i = -1, -1
    if start_i != -1:
        rowPairs.append((start_i, end_i))
    print(rowPairs)
    rowPairsArray = np.array(rowPairs)
    return (rowPairsArray)
#列分割
def get_vertical_seg_array(txt_pixels,len_y):
    start_i = -1
    end_i = -1
    rowPairs = []
    for i in range(len_y):
        if( txt_pixels[:,i].any() and start_i < 0):
            start_i = i
        elif( txt_pixels[:,i].any()):
            end_i = i
        elif ( not txt_pixels[:,i].any() and start_i >= 0):
            #print(end_i - start_i)
            if(end_i - start_i >= min_val):
                rowPairs.append((start_i, end_i))
                start_i, end_i = -1, -1
    if start_i != -1:
        rowPairs.append((start_i, end_i))
    print(rowPairs)
    rowPairsArray = np.array(rowPairs)
    return (rowPairsArray)

# for index, area in enumerate(rowPairsArray):
#     tmp_img = gray_img[area[0]:area[-1], :]
#     cv2.imwrite('seg' + str(index                                                                                                                                                                                                                                                                        ) + '.jpg', tmp_img)
# index = 1
# print (rowPairsArray)
# print (txt_pixels[50:60,])

#行分割写入
def horizontal_img_write(rowPairsArray,txt_pixels,label):
    for index,area in enumerate(rowPairsArray):
        # whiteArray = np.zeros((5,len_y))
        tmp_img = txt_pixels[area[0]:area[-1], :]
        # tmp_img = np.concatenate((whiteArray, tmp_img), axis=0)
        # tmp_img = np.concatenate((tmp_img, whiteArray), axis=0)
        cv2.imwrite('seg-result\\'+label[index]+ '.jpg', tmp_img)
#多字母分割
def mix_horizontal_img_write(rowPairsArray,txt_pixels,label,position):
    seg_content = []
    for index,area in enumerate(rowPairsArray):
        # whiteArray = np.zeros((5,len_y))
        seg_content.append(area)
        # tmp_img = np.concatenate((whiteArray, tmp_img), axis=0)
        # tmp_img = np.concatenate((tmp_img, whiteArray), axis=0)
    farea_x = seg_content[0][0]
    farea_y = seg_content[position-1][-1]
    larea_x = seg_content[position][0]
    larea_y = seg_content[-1][-1]
    # print(farea_x,farea_y)
    # print(larea_x, larea_y)
    if len_x > len_y:
        ftmp_img = txt_pixels[farea_x:farea_y,:]
        ltmp_img = txt_pixels[larea_x:larea_y,:]
        cv2.imwrite('mix-horizontal-result\\'+label[0:position]+'.jpg', ftmp_img)
        cv2.imwrite('mix-horizontal-result\\'+label[position:]+'.jpg',ltmp_img)
    else:
        ftmp_img = txt_pixels[:,farea_x:farea_y]
        ltmp_img = txt_pixels[:,larea_x:larea_y]
        cv2.imwrite('mix-vertical-result\\'+label[0:position] + '.jpg', ftmp_img)
        cv2.imwrite('mix-vertical-result\\'+label[position:] + '.jpg', ltmp_img)
#列分割写入
def vertical_img_write(rowPairsArray,txt_pixels,label):
    for index,area in enumerate(rowPairsArray):
        # whiteArray = np.zeros((len_x,5))
        tmp_img = txt_pixels[:,area[0]:area[-1]]
        # tmp_img = np.concatenate((whiteArray, tmp_img), axis=1)
        # tmp_img = np.concatenate((tmp_img, whiteArray), axis=1)
        cv2.imwrite('seg-result\\'+label[index]+ '.jpg', tmp_img)
def horizontal_vertical_write(txt_pixels,label):
    if len_x > len_y:
        rowPairsArray = get_horizontal_seg_array(txt_pixels,len_x)
        for index, area in enumerate(rowPairsArray):
            tmp_img = txt_pixels[area[0]:area[-1], :]
            rowPairsArray1 = get_vertical_seg_array(tmp_img,tmp_img.shape[1])
            cv2.imwrite('seg-result\\'+label[index]+ '.jpg',tmp_img[:,rowPairsArray1[0][0]:rowPairsArray1[0][1]])

    else:
        rowPairsArray = get_vertical_seg_array(txt_pixels,len_y)
        for index, area in enumerate(rowPairsArray):
            tmp_img = txt_pixels[:,area[0]:area[-1]]
            rowPairsArray1 = get_horizontal_seg_array(tmp_img,tmp_img.shape[0])
            cv2.imwrite('seg-result\\'+label[index]+ '.jpg',tmp_img[rowPairsArray1[0][0]:rowPairsArray1[0][1],:])

def main():
    horizontal_vertical_write(txt_pixels,label)
    # if len_x>len_y:
    #     rowPairsArray = get_horizontal_seg_array(txt_pixels,len_x)
    #     horizontal_img_write(rowPairsArray, txt_pixels,label)
    # else:
    #     rowPairsArray = get_vertical_seg_array(txt_pixels,len_y)
    #     vertical_img_write(rowPairsArray, txt_pixels,label)
    if len_x > len_y:
        rowPairsArray = get_horizontal_seg_array(txt_pixels,len_x)
        mix_horizontal_img_write(rowPairsArray,txt_pixels,label,position)
    else:
        rowPairsArray = get_vertical_seg_array(txt_pixels,len_y)
        mix_horizontal_img_write(rowPairsArray, txt_pixels, label, position)
if __name__ == '__main__':

    img = cv2.imread('mei.jpg')
    label = 'MATU513406'

    position = int(input("请输入分割点位置:"))
    if position>=1 and position <=len(label)-1:

        txt_pixels = get_txt_pixels(img)

        len_x = txt_pixels.shape[0]
        len_y = txt_pixels.shape[1]
        min_val = 2
        main()
    else:
        print("分割点不存在，请重新输入")