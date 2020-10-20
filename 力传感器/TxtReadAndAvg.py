import os
import xlrd
import xlwt
import numpy as np
import string

path = r'D:\1A.研究生\基于力的位姿计算\20201009采集'  # 待读取的文件夹
SavePath = r'D:\1A.研究生\基于力的位姿计算\20201009采集'

for folders in os.listdir(path):
    # read txt and average data in one txt
    txtPath=os.path.join(path,  folders)
    path_list = os.listdir(txtPath)
    SaveDir = os.path.join(SavePath,  folders, 'Avg.xls')
    path_list.sort()
    if 'Avg.xls' in path_list:
        path_list.remove('Avg.xls')
    print('path_list = ', path_list)
    Vol = []
    Vol_Force = []
    for filename in path_list:
        txt_dir = os.path.join(txtPath, filename)
        #print (txt_dir)
        fo = open(txt_dir, 'r')
        data = []
        dataForce = []
        for line in fo:
            line = line.strip('\n')  # very important 去除首尾空格
            line_ls = line.split(',')
            # data.append([float(f) for f in line_ls[14:20]])
            dataForce.append([float(f) for f in line_ls[21:27]])
        # Vol_Avg = np.mean(data, 0)
        Force_Avg = np.mean(dataForce, 0)
        #print ('Vol_Avg =',Vol_Avg)
        # Vol_Avg_list = Vol_Avg.tolist()         # type 'np.array' to type 'list'
        Force_Avg_list = Force_Avg.tolist()
        #print ('Vol_Avg_list =',Vol_Avg_list)
        # Vol_Avg_list.insert(0, filename)
        Force_Avg_list.insert(0, filename)
        # Vol.append(Vol_Avg_list)
        Vol_Force.append(Force_Avg_list)
        fo.close()
    # print('Vol = ', Vol)

    '''
    # D_value
    Vol_D_value = []
    m, n = np.shape(Vol)
    for i in range (m):
        D_value = []
        D_value.append(Vol[i][0])
        for j in range (6):
            D_value.append(Vol[i][j+1]-Vol[0][j+1])
        Vol_D_value.append(D_value)
    print ('Vol_D_value =', Vol_D_value )
    '''

    # Save
    Book = xlwt.Workbook(encoding='ascii', style_compression=0)
    # Sheet_1 = Book.add_sheet('VolAvg', cell_overwrite_ok=True)
    Sheet_1 = Book.add_sheet('ForceAvg', cell_overwrite_ok=True)
    a, b, = np.shape(Vol_Force)
    for i in range(a):
        for j in range(b):
            #Vol_Str = str(Vol[i][j])
            #Vol_Str = Vol[i][j]
            Force = Vol_Force[i][j]
            #Vol_D_value_Str = str(Vol_D_value[i][j])
            # Vol_D_value_Str = Vol_D_value[i][j]
            # Sheet.write(i,0,path_list[i])
            #Sheet_1.write(i,   j, Vol_Str)
            # Sheet.write(i+17,j, Vol_D_value_Str)
            Sheet_1.write(i, j, Force)
    Book.save(SaveDir)
