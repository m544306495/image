
import _random
import cv2
import numpy as np
from math  import  *
from sklearn.cluster import k_means
import  os
import  random
from PIL import  Image
#-----------------------
#   a,b:分割图片长度,宽度
#   fontcolor:字体颜色
#   x,y，bgcolor:背景图片长度,宽度，颜色
#   space: 分割图片间隔
#-----------------------

def to_gray(img):
    """
    转变成灰度图片
    :param img:
    :return:
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    cv2.imwrite('c' + str(1) + '.jpg', kmeans )
    # print(kmeans[0])
    return kmeans
#随机获得6个文件,返回file_names
def get_randomfile():
    rootdir = 'd:\\python\\program\\picture'
    file_names = []

    for parent, dirnames, filenames1 in os.walk(rootdir):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        filenames = filenames1
        for i in range(6):
            x = random.randint(0, len(filenames) - 1)
            file_names.append(filenames[x])
            i += 1
    return file_names
#接受文件列表，改变大小为30，50，存储在picture1中
def get_resize(file_names,tuple):
    for  filename in file_names:
        filenameContent = Image.open('picture\\{}'.format(filename))
        filenameContent1 = np.array(filenameContent.resize(tuple,Image.ANTIALIAS))
        filenameContent2 = cv2.medianBlur(filenameContent1,3)
        cv2.imwrite('picture\\{}'.format(filename),filenameContent2)
    return file_names


#获取背景图片修改开始坐标,返回start_lenx,start_leny
def get_horizontal_bgstart_pix(bgimg,space,a,b):
    bgimgarray = cv2.imread(bgimg)
    bglenx = bgimgarray.shape[0]
    bgleny = bgimgarray.shape[1]

    start_horizontal_lenx = int(ceil(abs((bglenx-b))/2))
    start_horizontal_leny = int(ceil(abs((bgleny-(6*a+5*space)))/2))
    return (start_horizontal_lenx,start_horizontal_leny)
def get_vertical_bgstart_pix(bgimg,space,a,b):
    bgimgarray = cv2.imread(bgimg)
    bglenx = bgimgarray.shape[0]
    bgleny = bgimgarray.shape[1]

    start_vertical_lenx = int(ceil(abs((bglenx-(6*b+5*space)))/2))
    start_vertical_leny = int(ceil(abs((bgleny-a))/2))
    print(start_vertical_lenx,start_vertical_leny)
    return (start_vertical_lenx, start_vertical_leny)
def horizontal_change(bgimg,change_file_names,start_horizontal_lenx,start_horizontal_leny,color,space,a,b):
    seg0 = get_txt_pixels(cv2.imread('picture\\{}'.format(change_file_names[0])))
    seg1 = get_txt_pixels(cv2.imread('picture\\{}'.format(change_file_names[1])))
    seg2 = get_txt_pixels(cv2.imread('picture\\{}'.format(change_file_names[2])))
    seg3 = get_txt_pixels(cv2.imread('picture\\{}'.format(change_file_names[3])))
    seg4 = get_txt_pixels(cv2.imread('picture\\{}'.format(change_file_names[4])))
    seg5 = get_txt_pixels(cv2.imread('picture\\{}'.format(change_file_names[5])))
    
    bgimgarray = cv2.imread(bgimg)
    print(seg0)
    try:
        for i in range(seg0.shape[0]):
            for j in range(seg0.shape[1]):
                try:
                    if seg0[i][j] != 0:
                        bgimgarray[i+start_horizontal_lenx][j+start_horizontal_leny]=color
                except:
                    continue
        for i in range(seg1.shape[0]):
            for j in range(seg1.shape[1]):
                try:
                    if seg1[i][j] != 0:
                        bgimgarray[i+start_horizontal_lenx][j+start_horizontal_leny+a+space]=color
                except:
                    continue
        for i in range(seg2.shape[0]):
            for j in range(seg2.shape[1]):
                try:
                    if seg2[i][j] != 0:
                        bgimgarray[i + start_horizontal_lenx][j + start_horizontal_leny + a*2 + space*2] = color
                except:
                    continue

        for i in range(seg3.shape[0]):
            for j in range(seg3.shape[1]):
                try:
                    if seg3[i][j] != 0:
                        bgimgarray[i + start_horizontal_lenx][j + start_horizontal_leny + a*3 + space*3] = color
                except:
                    continue
        for i in range(seg4.shape[0]):
            for j in range(seg4.shape[1]):
                try:
                    if seg4[i][j] != 0:
                        bgimgarray[i + start_horizontal_lenx][j + start_horizontal_leny + a*4 + space*4] = color
                except:
                    continue
        for i in range(seg5.shape[0]):
            for j in range(seg5.shape[1]):
                try:
                    if seg5[i][j] != 0:
                        bgimgarray[i + start_horizontal_lenx][j + start_horizontal_leny + a*5 + space*5] = color
                except:
                    continue
    except:
        pass
    finally:
        return bgimgarray
def vertical_change(bgimg,change_file_names,start_vertical_lenx,start_vertical_leny,color,space,a,b):
    seg0 = get_txt_pixels(cv2.imread('picture\\{}'.format(change_file_names[0])))
    seg1 = get_txt_pixels(cv2.imread('picture\\{}'.format(change_file_names[1])))
    seg2 = get_txt_pixels(cv2.imread('picture\\{}'.format(change_file_names[2])))
    seg3 = get_txt_pixels(cv2.imread('picture\\{}'.format(change_file_names[3])))
    seg4 = get_txt_pixels(cv2.imread('picture\\{}'.format(change_file_names[4])))
    seg5 = get_txt_pixels(cv2.imread('picture\\{}'.format(change_file_names[5])))
    bgimgarray1 = cv2.imread(bgimg)
    try:
        for i in range(seg0.shape[0]):
            for j in range(seg0.shape[1]):
                try:
                    if seg0[i][j] != 0:
                        bgimgarray1[i+start_vertical_lenx][j+start_vertical_leny]=color

                except:
                    continue
        for i in range(seg1.shape[0]):
            for j in range(seg1.shape[1]):
                try:
                    if seg1[i][j] != 0:
                        bgimgarray1[i+start_vertical_lenx+b+space][j+start_vertical_leny]=color
                except:
                    continue
        for i in range(seg2.shape[0]):
            for j in range(seg2.shape[1]):
                try:
                    if seg2[i][j] != 0:
                        bgimgarray1[i + start_vertical_lenx+ b*2 + space*2][j + start_vertical_leny ] = color
                except:
                    continue

        for i in range(seg3.shape[0]):
            for j in range(seg3.shape[1]):
                try:
                    if seg3[i][j] != 0:
                        bgimgarray1[i + start_vertical_lenx+ b*3 + space*3][j + start_vertical_leny] = color
                except:
                    continue
        for i in range(seg4.shape[0]):
            for j in range(seg4.shape[1]):
                try:
                    if seg4[i][j] != 0:
                        bgimgarray1[i + start_vertical_lenx+ b*4 + space*4][j + start_vertical_leny ] = color
                except:
                    continue
        for i in range(seg5.shape[0]):
            for j in range(seg5.shape[1]):
                try:
                    if seg5[i][j] != 0:
                        bgimgarray1[i + start_vertical_lenx+ b*5 + space*5][j + start_vertical_leny ] = color
                except:
                    continue
    except:
        pass
    finally:
        return bgimgarray1
def bgcolor_select(bgcolor):
    if bgcolor == "红色":
        bgcolor = "red"
    elif bgcolor == "绿色":
        bgcolor = "green"
    elif bgcolor == '黑色':
        bgcolor = "black"
    elif bgcolor == '灰色':
        bgcolor = "gray"
    elif bgcolor == '紫色':
        bgcolor = "purple"
    elif bgcolor == '黄色':
        bgcolor = "yellow"
    else:
        print("颜色不存在，默认背景为黑色")
        bgcolor = "black"
    return  (bgcolor)
def fontcolor_select(fontcolor):
    if fontcolor == '白色':
        color = 255
    elif fontcolor == '黑色':
        color = 0
    elif fontcolor == '灰色':
        color = 200
    else:
        print("颜色不存在，默认字体为白色")
        color = 255
    return  color
#文件结果递增
def result_name(change_file_names):
    path =  '{}{}{}{}{}{}.jpg'.format(change_file_names[0][0],change_file_names[1][0],change_file_names[2][0],change_file_names[3][0],change_file_names[4][0],change_file_names[5][0])
    exist_file = os.listdir('result')
    i=1
    while True:
        if path in exist_file:
            path = '{}{}{}{}{}{}-{}.jpg'.format(change_file_names[0][0],change_file_names[1][0],change_file_names[2][0],change_file_names[3][0],change_file_names[4][0],change_file_names[5][0],i)
            i+=1
        else:
            break
    print(path)
    return path
def select_bgimg(x,y,bgcolor):
        img_b = np.array(Image.new("RGB", (x, y),bgcolor))  # 不指定color，则为黑色#000000
        cv2.imwrite('{}.jpg'.format(bgcolor), img_b)
def main():
    # bgcolor = input('请输入背景图片颜色:\n')
    # x = int(input('请输入背景图片长度:\n'))
    # y = int(input('请输入背景图片宽度:\n'))
    # fontcolor = input('请输入字体颜色:\n')
    # a = int(input('请输入分割图片的长度:\n'))
    # b = int(input('请输入分割图片的宽度:\n'))
    # space = int(input('请输入分割图片的间距:\n'))
    bgcolor_list = ["黑色"]
    x_list = [900]
    y_list = [900]
    fontcolor_list = ['白色']
    a_list = [50]
    b_list = [30]
    space_list = [50]
    for aa in bgcolor_list:
        for bb in x_list:
            for cc in y_list:
                for dd in fontcolor_list:
                    for ee in a_list:
                        for ff in b_list:
                            for gg in space_list:
                                bgcolor = aa
                                x = bb
                                y = cc
                                fontcolor = dd
                                tuple = (ee,ff)
                                space = gg

                                bgcolor = bgcolor_select(bgcolor)

                                select_bgimg(x,y,bgcolor)
                                color = fontcolor_select(fontcolor)
                                file_names = get_randomfile()
                                change_file_names = get_resize(file_names,tuple)
                                start_horizontal_lenx,start_horizontal_leny = get_horizontal_bgstart_pix('{}.jpg'.format(bgcolor),space,ee,ff)
                                start_vertical_lenx,start_vertical_leny = get_vertical_bgstart_pix('{}.jpg'.format(bgcolor),space,ee,ff)
                                print(change_file_names)
                                path = result_name(change_file_names)
                                cv2.imwrite('result\\{}'.format(path),horizontal_change('{}.jpg'.format(bgcolor),change_file_names,start_horizontal_lenx,start_horizontal_leny,color,space,ee,ff))
                                path = result_name(change_file_names)
                                cv2.imwrite('result\\{}'.format(path),vertical_change('{}.jpg'.format(bgcolor), change_file_names,start_vertical_lenx, start_vertical_leny, color,space, ee, ff))


if __name__ == '__main__':
    main()






