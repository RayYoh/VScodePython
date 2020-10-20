import os
import xlrd
import xlwt
import math
import numpy as np
import scipy as sp  # 在numpy基础上实现的部分算法库
import matplotlib.pyplot as plt  # 绘图库
from scipy.optimize import leastsq  # 引入最小二乘法算法

data_path = r'D:\1A.研究生\基于力的位姿计算\20201009采集力值'  # 文件夹路径
RPY_angle = np.mat([[-0.1, 89.7, -9.3], [-105.5, 157.1, -108.1],
                    [28.6, 152.4, 25.1], [30.9, 131.9, 22.9],
                    [12.8, 126.1, -9.1], [52.1, 111.1, 14.1],
                    [43.4, 131.0, 21.7], [39.5, 125.9,
                                          10.7], [25.1, 126.7, 0.9],
                    [29.7, 136.1,
                     11.2]])  # 机器人姿态角（RPY）矩阵 #[-18.0, 179.8, -27.2],
RPY_angle = RPY_angle * math.pi / 180  # 转成弧度
print('姿态角矩阵[R P Y]：\n', RPY_angle)

row = RPY_angle.shape[0]  # 姿态个数

angle = RPY_angle
"""旋转矩阵"""
R_T = np.zeros((3 * row, 3))
for i in range(row):
    """根据公式计算"""
    R_T[3 * i, 0] = math.cos(angle[i, 0]) * math.cos(angle[i, 1]) * \
        math.cos(angle[i, 2]) - math.sin(angle[i, 0]) * math.sin(angle[i, 1])
    R_T[3 * i, 1] = math.sin(angle[i, 0]) * math.cos(angle[i, 1]) * \
        math.cos(angle[i, 2]) + math.cos(angle[i, 0]) * math.sin(angle[i, 2])
    R_T[3 * i, 2] = -math.sin(angle[i, 1]) * math.cos(angle[i, 2])
    R_T[3 * i + 1, 0] = -math.cos(angle[i, 0]) * \
        math.cos(angle[i, 1]) * math.sin(angle[i, 2])
    R_T[3 * i +
        1, 1] = -math.sin(angle[i, 0]) * math.cos(angle[i, 1]) * math.sin(
            angle[i, 2]) + math.cos(angle[i, 0]) * math.cos(angle[i, 2])
    R_T[3 * i + 1, 2] = math.sin(angle[i, 1]) * math.sin(angle[i, 2])
    R_T[3 * i + 2, 0] = math.cos(angle[i, 0]) * math.sin(angle[i, 1])
    R_T[3 * i + 2, 1] = math.sin(angle[i, 0]) * math.sin(angle[i, 1])
    R_T[3 * i + 2, 2] = math.cos(angle[i, 1])
print('R_T：\n', R_T)
"""读取实验数据"""
data_path_list = os.listdir(data_path)  # 读取文件夹
data_path_list.sort()  # 文件名排序

num = 0
data_path_list.sort()
Vol_all = []
for filename in data_path_list:
    xls_dir = os.path.join(data_path, filename)  # 表格路径
    print('表格：', filename)
    data = xlrd.open_workbook(xls_dir)  # 读取数据
    Sheet = data.sheet_by_index(0)  # 读取sheet
    Sheet_rows = Sheet.nrows  # 读取行数
    Vol_temp = []
    for i in range(1, Sheet_rows):
        data_str = Sheet.row_values(i)  # 按行读取
        list = [float(f) for f in data_str[1:]]
        Vol_temp.append(list)  # 读取数据
    Vol_all.append(Vol_temp)
    print("实验%d:\n" % num)
    Vol = np.mat(Vol_temp)
    print('Vol: \n', Vol)
    num += 1
    R_1_3 = []
    R_2_3 = []
    R_3_3 = []

    numTrain = 7

    for i in range(10):
        R_1_3.append(R_T[3 * i, 2])
        R_2_3.append(R_T[3 * i + 1, 2])
        R_3_3.append(R_T[3 * i + 2, 2])
    R = np.mat([R_1_3, R_2_3, R_3_3])
    print('R:\n', R)
    for i in range(3):
        x_1 = R[i, 0:numTrain].reshape(-1, 1)
        x_2 = np.square(x_1)
        I = np.mat([1 for _ in range(numTrain)]).reshape(-1, 1)
        x = np.hstack((x_2, x_1, I))
        y1 = Vol[0:numTrain, i]
        y2 = Vol[0:numTrain, i + 3]
        h1 = (np.transpose(x) * x).I * np.transpose(x) * y1.reshape(-1, 1)
        h2 = (np.transpose(x) * x).I * np.transpose(x) * y2.reshape(-1, 1)

        x_test_1 = R[i, numTrain:10].reshape(-1, 1)
        x_test_2 = np.square(x_test_1)
        I = np.mat([1 for _ in range(numTrain, 10)]).reshape(-1, 1)
        x_test = np.hstack((x_test_2, x_test_1, I))
        y1_test = Vol[numTrain:10, i]
        F_error = y1_test - np.dot(x_test, h1)
        y2_test = Vol[numTrain:10, i + 3]
        M_error = y2_test - np.dot(x_test, h2)
        if 0 == i:
            print('Fx = ', h1[0, 0], ' * R13^2 + ',
                  h1[1, 0], ' * R13 + ', h1[2, 0])
            print('Fx error= \n', F_error)
            print('Mx = ', h2[0, 0], ' * R13^2 + ',
                  h2[1, 0], ' * R13 + ', h2[2, 0])
            print('Mx error= \n', M_error)
        elif 1 == i:
            print('Fy = ', h1[0, 0], ' * R23^2 + ',
                  h1[1, 0], ' * R23 + ', h1[2, 0])
            print('Fy error= \n', F_error)
            print('My = ', h2[0, 0], ' * R23^2 + ',
                  h2[1, 0], ' * R23 + ', h2[2, 0])
            print('My error= \n', M_error)
        else:
            print('Fz = ', h1[0, 0], ' * R33^2 + ',
                  h1[1, 0], ' * R33 + ', h1[2, 0])
            print('Fz error= \n', F_error)

    y = Vol[0:numTrain, 5]
    I = np.mat([1 for _ in range(numTrain)]).reshape(-1, 1)
    x = np.hstack(
        (np.square(np.transpose(R[:, 0:numTrain])), np.transpose(R[:, 0:numTrain]), I))
    h = (np.transpose(x) * x).I * np.transpose(x) * y.reshape(-1, 1)
    print('Mz = ', h[0, 0], ' * R13^2 + ', h[1, 0], ' * R23^2 + ', h[2, 0], ' * R33^2 + ',
          h[3, 0], ' * R13 + ', h[4, 0], ' * R23 + ', h[5, 0], ' * R33 + ', h[6, 0])

    I = np.mat([1 for _ in range(numTrain, 10)]).reshape(-1, 1)
    x_test = np.hstack((np.square(np.transpose(
        R[:, numTrain:10])), np.transpose(R[:, numTrain:10]), I))
    y_test = Vol[numTrain:10, 5]
    Mz_error = y_test - np.dot(x_test, h)
    print('Mz error= \n', Mz_error)