import os
import xlrd
import xlwt
import math
import numpy as np
import matplotlib.pyplot as plt

data_path = r'D:\1A.研究生\基于力的位姿计算\2采重力补偿4'  # 文件夹路径
RPY_angle = np.mat([[-0.1, 89.7, -9.3], [-105.5, 157.1, -108.1],
                    [28.6, 152.4, 25.1], [30.9, 131.9, 22.9],
                    [12.8, 126.1,
                     -9.1]])  # 机器人姿态角（RPY）矩阵 #[-18.0, 179.8, -27.2],
RPY_angle = RPY_angle * math.pi / 180  # 转成弧度
print('姿态角矩阵[R P Y]：\n', RPY_angle)

row = RPY_angle.shape[0]  # 姿态个数
Rxyz = np.zeros((3 * row, 3))  # 初始化矩阵
angle = np.zeros((row, 3))
'''
for i in range(row):
    """根据公式计算 旋转矩阵"""
    Rxyz[3 * i, 0] = math.cos(RPY_angle[i, 0]) * math.cos(RPY_angle[i, 1])
    Rxyz[3 * i, 1] = math.cos(RPY_angle[i, 0]) * math.sin(RPY_angle[i, 1]) * math.sin(
        RPY_angle[i, 2]) - math.sin(RPY_angle[i, 0]) * math.cos(RPY_angle[i, 2])
    Rxyz[3 * i, 2] = math.cos(RPY_angle[i, 0]) * math.sin(RPY_angle[i, 1]) * math.cos(
        RPY_angle[i, 2]) + math.sin(RPY_angle[i, 0]) * math.cos(RPY_angle[i, 2])
    Rxyz[3 * i + 1, 0] = math.sin(RPY_angle[i, 0]) * math.cos(RPY_angle[i, 1])
    Rxyz[3 * i + 1, 1] = math.sin(RPY_angle[i, 0]) * math.sin(RPY_angle[i, 1]) * math.sin(
        RPY_angle[i, 2]) + math.cos(RPY_angle[i, 0]) * math.cos(RPY_angle[i, 2])
    Rxyz[3 * i + 1, 2] = math.sin(RPY_angle[i, 0]) * math.sin(RPY_angle[i, 1]) * math.cos(
        RPY_angle[i, 2]) - math.cos(RPY_angle[i, 0]) * math.sin(RPY_angle[i, 2])
    Rxyz[3 * i + 2, 0] = - math.sin(RPY_angle[i, 1])
    Rxyz[3 * i + 2, 1] = math.cos(RPY_angle[i, 1]) * math.sin(RPY_angle[i, 2])
    Rxyz[3 * i + 2, 2] = math.cos(RPY_angle[i, 1]) * math.cos(RPY_angle[i, 2])
print('Rxyz: ', Rxyz)
R_T = np.zeros((3*row, 3))
for i in range(row):
    R_T[3*i:3*(i+1), :] = np.transpose(Rxyz[3*i:3*(i+1), :])
print('R_T:\n', R_T)
'''
'''
for i in range(row):

    R = Rxyz[3 * i:3 * i + 3, :]  # 取第i个姿态的旋转矩阵
    print(R)
    sinB = math.sqrt(R[2, 0] ** 2 + R[2, 1] ** 2)

    if abs(sinB) < 0.00001:  # sinB=0的情况
        angle[i, 0] = 0
        angle[i, 1] = 0
        angle[i, 2] = math.atan2(-R[0, 1], R[0, 0])
    else:
        angle[i, 1] = math.atan2(sinB, R[2, 2])
        angle[i, 0] = math.atan2(R[1, 2], R[0, 2])
        angle[i, 2] = math.atan2(R[2, 1], -R[2, 0])

# angle =
# np.matrix([[0,120,180],[-55,167.7,55.5],[101.4,156.1,-57.9],[60.1,130,-90],[72,147.6,-63.8],[139,167.7,-40.5]])
# #机器人姿态角矩阵
# angle = angle * math.pi / 180 #转成弧度
print('姿态角矩阵[A B C]：\n', angle)
print('姿态角矩阵[A B C](角度)：\n', angle * 180 / math.pi)
# row = angle.shape[0] #姿态个数
'''

angle = RPY_angle
"""旋转矩阵"""
R_T = np.zeros((3 * row, 3))
for i in range(row):
    """根据公式计算"""
    R_T[3 * i, 0] = math.cos(angle[i, 0]) * math.cos(angle[i, 1]) * \
        math.cos(angle[i, 2]) - math.sin(angle[i, 0]) * math.sin(angle[i, 1])
    R_T[3 * i, 1] = math.sin(angle[i, 0]) * math.cos(angle[i, 1]) * \
        math.cos(angle[i, 2]) + math.cos(angle[i, 0]) * math.sin(angle[i, 2])
    R_T[3 * i, 2] = -math.sin(angle[i, 0]) * math.cos(angle[i, 2])
    R_T[3 * i + 1, 0] = -math.cos(angle[i, 0]) * \
        math.cos(angle[i, 1]) * math.sin(angle[i, 2])
    R_T[3 * i +
        1, 1] = -math.sin(angle[i, 0]) * math.cos(angle[i, 1]) * math.sin(
            angle[i, 2]) + math.cos(angle[i, 0]) * math.cos(angle[i, 2])
    R_T[3 * i + 1, 2] = math.sin(angle[i, 1]) * math.sin(angle[i, 2])
    R_T[3 * i + 2, 0] = math.cos(angle[i, 0]) * math.sin(angle[i, 1])
    R_T[3 * i + 2, 1] = math.sin(angle[i, 0]) * math.sin(angle[i, 1])
    R_T[3 * i + 2, 2] = math.cos(angle[i, 1])
print('拼接前R：\n', R_T)

I = np.eye(3)  # 创建单位矩阵
for i in range(row - 1):
    I = np.vstack((I, np.eye(3)))  # 拼接

# print(I)
R = np.hstack((R_T, I))  # 矩阵拼接
R = np.mat(R)
print('拼接后R：\n', R)
"""读取实验数据"""
data_path_list = os.listdir(data_path)  # 读取文件夹
data_path_list.sort()  # 文件名排序

G_list = []
alpha_list = []
belta_list = []
F_M_0_list = []
num = 0
F_M_0_all = np.zeros((len(data_path_list), 6))
x_y_z = np.zeros((len(data_path_list), 3))
data_path_list.sort()
Vol_all = []
for filename in data_path_list:
    xls_dir = os.path.join(data_path, filename)  # 表格路径
    print('表格：', filename)
    data = xlrd.open_workbook(xls_dir)  # 读取数据
    Sheet = data.sheet_by_index(0)  # 读取sheet
    Sheet_rows = Sheet.nrows  # 读取行数
    Vol_temp = []
    for i in range(row):
        data_str = Sheet.row_values(i)  # 按行读取
        list = [float(f) for f in data_str[1:]]
        Vol_temp.append(list)  # 读取数据
    Vol_all.append(Vol_temp)
    print("实验%d:\n" % num)
    Vol = np.mat(Vol_temp)
    print('Vol: ', Vol)

    T_temp = np.zeros((3 * row, 3))  # 初始化矩阵
    for i in range(row):

        T_temp[3 * i, 1] = Vol[i, 2]
        T_temp[3 * i, 2] = -Vol[i, 1]
        T_temp[3 * i + 1, 0] = -Vol[i, 2]
        T_temp[3 * i + 1, 2] = Vol[i, 0]
        T_temp[3 * i + 2, 0] = Vol[i, 1]
        T_temp[3 * i + 2, 1] = -Vol[i, 0]

    print('拼接前T：\n', T_temp)

    T = np.hstack((T_temp, I))
    T = np.mat(T)
    print('拼接后T：\n', T)
    """分别计算力/力矩矩阵"""
    F = np.zeros((3 * row, 1))
    M = np.zeros((3 * row, 1))
    # print(F_M.shape)

    for j in range(row):

        F_T = Vol[j, 0:3]
        M_T = Vol[j, 3:6]

        F[3 * j:3 * (j + 1), :] = np.transpose(F_T)  # 转成列矩阵
        M[3 * j:3 * (j + 1), :] = np.transpose(M_T)

    F = np.mat(F)
    M = np.mat(M)
    print('力F：\n', F)
    print('力矩M：\n', M)
    """最小二乘法计算"""
    Mat_R = np.transpose(R) * R
    Mat_R_I = Mat_R.I
    h = Mat_R_I * np.transpose(R) * F
    # print(h)

    G = math.sqrt(h[0, 0]**2 + h[1, 0]**2 + h[2, 0]**2)  # 计算重力
    print('重力G：\n', G)

    alpha = math.asin(-h[1, 0] / G) * 180 / math.pi  # 计算两个角度
    belta = math.atan(-h[0, 0] / h[2, 0]) * 180 / math.pi

    print('alpha:\n', alpha)
    print('belta:\n', belta)

    # F_0=np.zeros((3,1))
    F_0 = h[3:6, :]  # 力零点
    # print('力零点F0:\n',F_0)
    Mat_T = np.transpose(T) * T
    Mat_T_I = Mat_T.I
    m = Mat_T_I * np.transpose(T) * M

    print('重心坐标:\n', m[0:3, :])

    T_0 = np.mat([[0, F_0[2, 0], -F_0[1, 0], 1, 0, 0],
                  [-F_0[2, 0], 0, F_0[0, 0], 0, 1, 0],
                  [F_0[1, 0], -F_0[0, 0], 0, 0, 0, 1]])
    # T_0=np.mat(T_0)
    M_0 = T_0 * m  # 力矩零点
    # print('T_0:\n',T_0)
    # print('力矩零点M0：\n',M_0)
    F_M_0 = np.vstack((F_0, M_0))
    print('传感器零点：\n', F_M_0)
    # print(F_M_0.tolist())

    G_list.append(G)
    alpha_list.append(alpha)
    belta_list.append(belta)
    F_M_0_all[num, :] = np.transpose(F_M_0)
    x_y_z[num, :] = np.transpose(m[0:3, :])
    num = num + 1

print('文件:\n', data_path_list)
print('G:\n', G_list)
print('Alpha:\n', alpha_list)
print('Belta:\n', belta_list)
print('重心：\n', x_y_z)
print('F_M_0:\n', F_M_0_all)
F_M_0_avg = np.sum(F_M_0_all, axis=0) / 6
alpha = np.mean(alpha_list[1:4])
belta = np.mean(belta_list[1:4])

R_1_T = R_T[0:3, :]
gx = G_list[0] * math.cos(alpha) * math.sin(belta)
gy = -G_list[0] * math.sin(alpha)
gz = -G_list[0] * math.cos(alpha) * math.cos(belta)
G_0 = np.dot(R_1_T, np.mat([[gx], [gy], [gz]]))
print('G_0:\n', G_0)
Mg_x = gz * x_y_z[0, 1] - gy * x_y_z[0, 2]
Mg_y = gx * x_y_z[0, 2] - gz * x_y_z[0, 0]
Mg_z = gy * x_y_z[0, 0] - gx * x_y_z[0, 1]
Mg_0 = np.mat([[Mg_x], [Mg_y], [Mg_z]])

G_Mg_0 = np.vstack((G_0, Mg_0))
G_true_all = []
print('Vol_all:\n', Vol_all)
for i in range(1,4):
    G_sum = 0
    for j in range(5):
        F_test = np.transpose(np.mat(Vol_all[i])[j, :])
        f = F_test - G_Mg_0 - np.transpose(F_M_0_avg).reshape(-1, 1)
        G_true = math.sqrt(f[0, :]**2 + f[1, :]**2 + h[2, :]**2)
        G_sum += G_true
    G_true_all.append(G_sum / 5)
print('G_true_all:\n', G_true_all)
