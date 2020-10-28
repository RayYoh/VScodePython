import os
import xlrd
import xlwt
import math
import numpy as np
import scipy as sp  # 在numpy基础上实现的部分算法库
import matplotlib.pyplot as plt
from scipy.optimize import leastsq  # leastsq

"""Read experimental data"""


def readData(path):
    '''Read experimental data.

    Args:
    path: xls dir

    Returns:
    vol_Force: experimental force
    RPY_angle: RPY angle matrix of robot
    '''
    data = xlrd.open_workbook(path)  # read excel
    Sheet = data.sheet_by_index(0)  # read sheet
    Sheet_rows = Sheet.nrows  # read row
    vol_Force = []
    RPY_angle = []
    for i in range(0, Sheet_rows):
        data_str = Sheet.row_values(i)  # read by row
        force_list = [float(f) for f in data_str[4:10]]
        vol_Force.append(force_list)  # read force
        angle_list = [float(a) for a in data_str[19:22]]
        RPY_angle.append(angle_list)  # read attitude angle

    vol_Force = np.mat(vol_Force)
    RPY_angle = np.mat(RPY_angle)  # RPY angle matrix
    RPY_angle = RPY_angle * math.pi / 180  # 。to rad
    
    return vol_Force, RPY_angle


"""Calculate the transpose rotation matrix"""


def CalRatMat(RPY_angle):
    '''Calculate the transpose rotation matrix

    Args:
    RPY_angle: RPY angle matrix of robot

    Returns:
    R_T: transpose matrix of rotation matrix
    cal_R: A matrix of spliced by all R13,R23,R33 
    '''
    row = RPY_angle.shape[0]  # number of attitute

    angle = RPY_angle
    """rotation matrix"""
    R_T = np.zeros((3 * row, 3))
    for i in range(row):
        """calculate by formula"""
        R_T[3 * i, 0] = math.cos(angle[i, 0]) * math.cos(angle[i, 1]) * \
            math.cos(angle[i, 2]) - math.sin(angle[i, 0]) * \
                     math.sin(angle[i, 1])
        R_T[3 * i, 1] = math.sin(angle[i, 0]) * math.cos(angle[i, 1]) * \
            math.cos(angle[i, 2]) + math.cos(angle[i, 0]) * \
                     math.sin(angle[i, 2])
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
    # print('R_T：\n', R_T)
    R_1_3 = []
    R_2_3 = []
    R_3_3 = []

    for i in range(row):
        R_1_3.append(R_T[3 * i, 2])
        R_2_3.append(R_T[3 * i + 1, 2])
        R_3_3.append(R_T[3 * i + 2, 2])
    cal_R = np.mat([R_1_3, R_2_3, R_3_3])  # Splice R13,R23,R33 into a matrix
    return R_T, cal_R


"""Train quadratic function"""


def train_Qua(vol_Force, cal_R, numTrain):
    '''Train data by quadratic function

    Args:
    vol_Force: experimental force data
    cal_R: A matrix of spliced by all R13,R23,R33 
    numTrain: Number of training set data
    '''
    row = vol_Force.shape[0]
    for i in range(3):
        x_1 = cal_R[i, 0:numTrain].reshape(-1, 1)
        x_2 = np.square(x_1)
        I = np.mat([1 for _ in range(numTrain)]).reshape(-1, 1)
        x = np.hstack((x_2, x_1, I))
        y1 = vol_Force[0:numTrain, i]
        y2 = vol_Force[0:numTrain, i + 3]
        h1 = (np.transpose(x) * x).I * np.transpose(x) * y1.reshape(-1, 1)
        h2 = (np.transpose(x) * x).I * np.transpose(x) * y2.reshape(-1, 1)

        x_test_1 = cal_R[i, numTrain:row].reshape(-1, 1)
        x_test_2 = np.square(x_test_1)
        I = np.mat([1 for _ in range(numTrain, row)]).reshape(-1, 1)
        x_test = np.hstack((x_test_2, x_test_1, I))
        y1_test = vol_Force[numTrain:row, i]
        F_error = y1_test - np.dot(x_test, h1)
        y2_test = vol_Force[numTrain:row, i + 3]
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

    y = vol_Force[0:numTrain, 5]
    I = np.mat([1 for _ in range(numTrain)]).reshape(-1, 1)
    x = np.hstack(
        (np.square(np.transpose(cal_R[:, 0:numTrain])), np.transpose(cal_R[:, 0:numTrain]), I))
    h = (np.transpose(x) * x).I * np.transpose(x) * y.reshape(-1, 1)
    print('Mz = ', h[0, 0], ' * R13^2 + ', h[1, 0], ' * R23^2 + ', h[2, 0], ' * R33^2 + ',
          h[3, 0], ' * R13 + ', h[4, 0], ' * R23 + ', h[5, 0], ' * R33 + ', h[6, 0])

    I = np.mat([1 for _ in range(numTrain, row)]).reshape(-1, 1)
    x_test = np.hstack((np.square(np.transpose(
        cal_R[:, numTrain:row])), np.transpose(cal_R[:, numTrain:row]), I))
    y_test = vol_Force[numTrain:row, 5]
    Mz_error = y_test - np.dot(x_test, h)
    print('Mz error= \n', Mz_error)
    return


"""Train linear function"""


def train_Lin(vol_Force, cal_R, numTrain):
    '''Train data by linear function

    Args:
    vol_Force: experimental force data
    cal_R: A matrix of spliced by all R13,R23,R33 
    numTrain: Number of training set data
    '''
    row = vol_Force.shape[0]
    for i in range(3):
        x_1 = cal_R[i, 0:numTrain].reshape(-1, 1)
        I = np.mat([1 for _ in range(numTrain)]).reshape(-1, 1)
        x = np.hstack((x_1, I))
        y1 = vol_Force[0:numTrain, i]
        y2 = vol_Force[0:numTrain, i + 3]
        h1 = (np.transpose(x) * x).I * np.transpose(x) * y1.reshape(-1, 1)
        h2 = (np.transpose(x) * x).I * np.transpose(x) * y2.reshape(-1, 1)

        x_test_1 = cal_R[i, numTrain:row].reshape(-1, 1)
        I = np.mat([1 for _ in range(numTrain, row)]).reshape(-1, 1)
        x_test = np.hstack((x_test_1, I))
        y1_test = vol_Force[numTrain:row, i]
        F_error = y1_test - np.dot(x_test, h1)
        y2_test = vol_Force[numTrain:row, i + 3]
        M_error = y2_test - np.dot(x_test, h2)
        if 0 == i:
            print('Fx = ', h1[0, 0], ' * R13 + ',
                  h1[1, 0])
            print('Fx error= \n', F_error)
            print('Mx = ', h2[0, 0], ' * R13 + ',
                  h2[1, 0],)
            print('Mx error= \n', M_error)
        elif 1 == i:
            print('Fy = ', h1[0, 0], ' * R23 + ',
                  h1[1, 0])
            print('Fy error= \n', F_error)
            print('My = ', h2[0, 0], ' * R23 + ',
                  h2[1, 0])
            print('My error= \n', M_error)
        else:
            print('Fz = ', h1[0, 0], ' * R33 + ',
                  h1[1, 0])
            print('Fz error= \n', F_error)

    y = vol_Force[0:numTrain, 5]
    I = np.mat([1 for _ in range(numTrain)]).reshape(-1, 1)
    x = np.hstack(
        (np.transpose(cal_R[:, 0:numTrain]), I))
    h = (np.transpose(x) * x).I * np.transpose(x) * y.reshape(-1, 1)
    print('Mz = ', h[0, 0], ' * R13 + ', h[1, 0], ' * R23 + ', h[2, 0], ' * R33 + ',
          h[3, 0])

    I = np.mat([1 for _ in range(numTrain, row)]).reshape(-1, 1)
    x_test = np.hstack((np.transpose(cal_R[:, numTrain:row]), I))
    y_test=vol_Force[numTrain:row, 5]
    Mz_error=y_test - np.dot(x_test, h)
    print('Mz error= \n', Mz_error)
    
if __name__ == '__main__':
    data_path=r'D:\1A.研究生\科研\基于力的位姿计算\不同姿态下的传感器输出值.xlsx'  # xls dir
    vol_Force, RPY_angle=readData(data_path)
    R_T, cal_R=CalRatMat(RPY_angle)
    train_Lin(vol_Force, cal_R, 15400)
