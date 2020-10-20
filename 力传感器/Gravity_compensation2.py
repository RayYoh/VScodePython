import os
import xlrd
import xlwt
import math
import numpy as np
import matplotlib.pyplot as plt

data_path = r'D:\1A.研究生\基于力的位姿计算\重力补偿2'   # 文件夹路径
# Gc_path = "D:/毕设材料/20191227标定数据处理/计算结果/Python（线性）计算结果.xls" #读取标定矩阵Gc
RPY_angle = np.mat([[-0.1, 89.7, -9.3], [-18.0, 179.8, -27.2], [-105.5, 157.1, -108.1],
                    [28.6, 152.4, 25.1], [30.9, 131.9, 22.9], [12.8, 126.1, -9.1]])  # 机器人姿态角（RPY）矩阵
RPY_angle = RPY_angle * math.pi / 180  # 转成弧度
print('姿态角矩阵[R P Y]：\n', RPY_angle)

row = RPY_angle.shape[0]  # 姿态个数
Rxyz = np.zeros((3 * row, 3))  # 初始化矩阵
angle = np.zeros((row, 3))

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
    R_T[3  *i:3 * (i+1), :] = np.transpose(Rxyz[3 * i:3 * (i+1), :])
print('R_T:\n', R_T)

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
'''
"""读取标定矩阵Gc"""
Gc_temp = []
Gc_data = xlrd.open_workbook(Gc_path)  # 打开xls文件
Sheet = Gc_data.sheet_by_index(0)
Sheet_rows = Sheet.nrows  # 读取行数
for i in range(Sheet_rows):
    Gc_data_str = Sheet.row_values(i)  # 按行读取
    list = [float(f) for f in Gc_data_str]
    Gc_temp.append(list)  # 读取数据
Gc = np.mat(Gc_temp)  # 转成矩阵
print('标定矩阵Gc:\n', Gc)
'''
'''
angle=RPY_angle
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
    R_T[3 * i + 1, 1] = -math.sin(angle[i, 0]) * math.cos(angle[i, 1]) * math.sin(
        angle[i, 2]) + math.cos(angle[i, 0]) * math.cos(angle[i, 2])
    R_T[3 * i + 1, 2] = math.sin(angle[i, 1]) * math.sin(angle[i, 2])
    R_T[3 * i + 2, 0] = math.cos(angle[i, 0]) * math.sin(angle[i, 1])
    R_T[3 * i + 2, 1] = math.sin(angle[i, 0]) * math.sin(angle[i, 1])
    R_T[3 * i + 2, 2] = math.cos(angle[i, 1])
print('拼接前R：\n', R_T)
'''
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
Vol_all=[]
for filename in data_path_list:
    xls_dir = os.path.join(data_path, filename)  # 表格路径
    data = xlrd.open_workbook(xls_dir)  # 读取数据
    Sheet = data.sheets()[1]  # 读取sheet
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
    '''
    Vol_temp_mat = np.mat(Vol_temp)  # 转成矩阵
    Vol = np.transpose(Vol_temp_mat)  # 转置
    print('实验数据：\n', Vol)
    print(Vol.shape)
    F_M = Gc * Vol  # 计算力/力矩
    print('计算的F/M：\n', F_M)
    '''

    T_temp = np.zeros((3 * row, 3))  # 初始化矩阵
    for i in range(row):
        '''
        T_temp[3 * i, 1] = F_M[2, i]
        T_temp[3 * i, 2] = -F_M[1, i]
        T_temp[3 * i + 1, 0] = -F_M[2, i]
        T_temp[3 * i + 1, 2] = F_M[0, i]
        T_temp[3 * i + 2, 0] = F_M[1, i]
        T_temp[3 * i + 2, 1] = -F_M[0, i]
        '''
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
        '''
        list1 = [float(f) for f in Vol[j, 0:3].tolist()[0]]  # 读取力
        list2 = [float(f) for f in Vol[j, 3:6].tolist()[0]]  # 读取力矩
        # F_list.append(list)
        '''
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

    G = math.sqrt(h[0, 0] ** 2 + h[1, 0] ** 2 + h[2, 0] ** 2)  # 计算重力
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

    T_0 = np.mat([[0, F_0[2, 0], -F_0[1, 0], 1, 0, 0], [-F_0[2, 0],
                                                        0, F_0[0, 0], 0, 1, 0], [F_0[1, 0], -F_0[0, 0], 0, 0, 0, 1]])
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
    '''
    G_x_y_z = R_T[3:6, :] * h[0:3, 0]
    print("G_x_y_z：\n", G_x_y_z)

    mg = np.mat([[0, -m[2, 0], m[1, 0]],
                 [m[2, 0], 0, -m[0, 0]], [-m[1, 0], m[0, 0], 0]])
    Mg_x_y_z = mg * G_x_y_z[0:3, 0]
    print("Mg_x_y_z:\n", Mg_x_y_z)

    F_ = F[3:6, :] - F_0[0:3, :] - G_x_y_z[0:3, :]
    M_ = M[3:6, :] - M_0[0:3, :] - Mg_x_y_z[0:3, :]
    print("F_", F_)
    print("M_", M_)
    Fe = math.sqrt(F_[0, 0]**2 + F_[1, 0]**2 + F_[2, 0] ** 2)
    Me = math.sqrt(M_[0, 0]**2 + M_[1, 0]**2 + M_[2, 0] ** 2)
    print("Fe", Fe)
    print("Me", Me)
    '''


print('G:\n', G_list)
print('Alpha:\n', alpha_list)
print('Belta:\n', belta_list)
print('重心：\n', x_y_z)
print('F_M_0:\n', F_M_0_all)

R_1_T = R_T[0:3, :]
gx = G_list[0]*math.cos(alpha_list[0])*math.sin(belta_list[0])
gy = -G_list[0]*math.sin(alpha_list[0])
gz = -G_list[0]*math.cos(alpha_list[0])*math.cos(belta_list[0])
G_0 = np.dot(R_1_T,np.mat([[gx], [gy], [gz]]))
print('G_0:\n',G_0)
Mg_x = gz*x_y_z[0, 1]-gy*x_y_z[0, 2]
Mg_y = gx*x_y_z[0, 2]-gz*x_y_z[0, 0]
Mg_z = gy*x_y_z[0, 0]-gx*x_y_z[0, 1]
Mg_0 = np.mat([[Mg_x], [Mg_y], [Mg_z]])

G_Mg_0=np.vstack((G_0,Mg_0))
G_true_all=[]
for i in range(4):
    G_sum=0
    for j in range(6):  
        F_test=np.transpose(np.mat(Vol_all[i])[j,:])
        f=F_test-G_Mg_0-np.transpose(F_M_0_all[i,:]).reshape(-1,1)
        G_true=math.sqrt(f[0, :] ** 2 + f[1, :] ** 2 + h[2, :] ** 2)
        G_sum+=G_true
    G_true_all.append(G_sum/6)
print('G_true_all:\n',G_true_all)

'''
# 画重力
plt.scatter(n,G_list)
plt.title("Load gravity")
plt.xlabel("Number")
plt.ylabel("Gravity/N")
#plt.savefig("D:/毕设材料/论文/重力.png")
plt.show()



fig=plt.figure()
y0=x_y_z[:,0]
y1=x_y_z[:,1]
y2=x_y_z[:,2]
plt.subplot(311)
plt.plot(n,y0,'-ro')
plt.title("Gravity coordinates")
plt.ylabel("x/cm")
plt.subplot(312)
plt.plot(n,y1,'--bo')
plt.ylabel("y/cm")
plt.subplot(313)
plt.plot(n,y2,'-.go')
plt.xlabel("Number")
plt.ylabel("z/cm")
# plt.savefig("D:/毕设材料/论文/重力坐标.png")
plt.show()
'''
