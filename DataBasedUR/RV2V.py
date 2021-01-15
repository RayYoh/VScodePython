import xlrd
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import util
from sklearn import preprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



""" Read experimental data """
# For UR robot
def readDataAll(path):
    '''For all Data'''
    data = xlrd.open_workbook(path)  # read excel
    Sheet = data.sheet_by_index(0)  # read sheet
    Sheet_rows = Sheet.nrows  # read row
    volList = []
    for i in range(0, Sheet_rows):
        dataLine = Sheet.row_values(i)
        rawTraForce = [f for f in dataLine[15:21]]
        rawConForce = [f for f in dataLine[35:41]]
        rawTCPPos = [f for f in dataLine[42:48]]
        rawRPY = util.rv2rpy(rawTCPPos[3], rawTCPPos[4], rawTCPPos[5])  # Rotation Vector 2 RPY
        rawV = [f for f in dataLine[28:34]]
        volList.append(rawTraForce + rawConForce + rawTCPPos + list(rawRPY) + rawV)
    vol = np.mat(volList)
    # np.random.shuffle(vol)
    traForce = vol[:, :6]
    conForce = vol[:, 6:12]
    TCPRv = vol[:, 15:18]
    RPYangle = vol[:, 18:21]
    V = vol[:,21:]
    return traForce,conForce,TCPRv,RPYangle,V

def readData(path):
    '''Read experimental grouped data.

    Args:
    path: xls dir

    Returns:
    vol_Force: experimental force
    RPY_angle: RPY angle matrix of robot
    '''
    data = xlrd.open_workbook(path)  # read excel
    Sheet = data.sheet_by_index(0)  # read sheet
    Sheet_rows = Sheet.nrows  # read row
    volList = []
    for i in range(0, Sheet_rows):
        dataLine = Sheet.row_values(i)
        rawConForce = [f for f in dataLine[0:6]]
        rawTCPRv = [f for f in dataLine[12:]]
        rawRPY = util.rv2rpy(rawTCPRv[0], rawTCPRv[1], rawTCPRv[2])  # Rotation Vector 2 RPY
        rawV = [f for f in dataLine[6:12]]
        volList.append(rawConForce+rawTCPRv+list(rawRPY)+rawV)
    vol = np.mat(volList)
    conForce = vol[:, :6]
    TCPRv = vol[:, 6:9]
    RPYangle = vol[:, 9:12]
    V = vol[:, 12:]

    return conForce,TCPRv, RPYangle, V

def RV2RM(TCPRv):
    row = np.shape(TCPRv)[0]
    RM = []
    for i in range(row):
        RM.append(util.rv2rm(TCPRv[i,0],TCPRv[i,1],TCPRv[i,2]))
    # RM = np.array(RM)
    return RM
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
        length = len(list(F_error))
        if 0 == i:
            print('Fx = ', h1[0, 0], ' * R13^2 + ',
                  h1[1, 0], ' * R13 + ', h1[2, 0])
            print('Fx error= \n', F_error)
            plt.scatter([i for i in range(length)], list(F_error))
            plt.ylim(-10, 10)
            plt.show()
            print('Mx = ', h2[0, 0], ' * R13^2 + ',
                  h2[1, 0], ' * R13 + ', h2[2, 0])
            print('Mx error= \n', M_error)
            plt.scatter([i for i in range(length)], list(M_error))
            plt.show()
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
    '''Fx,Fy,Fz'''
    for i in range(3):
        x_1 = cal_R[i, 0:numTrain].reshape(-1, 1)
        I = np.mat([1 for _ in range(numTrain)]).reshape(-1, 1)
        x = np.hstack((x_1, I))
        y = vol_Force[0:numTrain, i]
        h = (np.transpose(x) * x).I * np.transpose(x) * y.reshape(-1, 1)

        x_test_1 = cal_R[i, numTrain:row].reshape(-1, 1)
        I = np.mat([1 for _ in range(numTrain, row)]).reshape(-1, 1)
        x_test = np.hstack((x_test_1, I))
        y_test = vol_Force[numTrain:row, i]
        F_error = y_test - np.dot(x_test, h)
        length = len(list(F_error))
        if 0 == i:
            # print('Fx = ', h[0, 0], ' * R13 + ',
            #       h[1, 0])
            # print('Fx error= \n', F_error)
            plt.scatter([i for i in range(length)], list(F_error))
            plt.ylim(-10, 10)
            plt.title('Linear Fx_error')
            plt.show()
        elif 1 == i:
            # print('Fy = ', h[0, 0], ' * R23 + ',
            #       h[1, 0])
            # print('Fy error= \n', F_error)
            plt.scatter([i for i in range(length)], list(F_error))
            plt.ylim(-10, 10)
            plt.title('Linear Fy_error')
            plt.show()
        else:
            # print('Fz = ', h[0, 0], ' * R33 + ',
            #       h[1, 0])
            # print('Fz error= \n', F_error)
            plt.scatter([i for i in range(length)], list(F_error))
            plt.ylim(-10, 10)
            plt.title('Linear Fz_error')
            plt.show()

    '''Mx'''
    I = np.mat([1 for _ in range(numTrain)]).reshape(-1, 1)
    x = np.hstack(
        (np.transpose(cal_R[:, 0:numTrain]), I))
    y = vol_Force[0:numTrain, 3]
    h = (np.transpose(x) * x).I * np.transpose(x) * y.reshape(-1, 1)
    # print('Mz = ', h[0, 0], ' * R13 + ', h[1, 0], ' * R23 + ', h[2, 0], ' * R33 + ',
    #       h[3, 0])

    I = np.mat([1 for _ in range(numTrain, row)]).reshape(-1, 1)
    x_test = np.hstack((np.transpose(cal_R[:, numTrain:row]), I))
    y_test = vol_Force[numTrain:row, 3]
    M_error = y_test - np.dot(x_test, h)
    # print('Mz error= \n', Mz_error)
    length = len(list(M_error))
    plt.scatter([i for i in range(length)], list(M_error))
    # plt.ylim(-10, 10)
    plt.title('Linear Mx_error')
    plt.show()

    '''My'''
    I = np.mat([1 for _ in range(numTrain)]).reshape(-1, 1)
    x = np.hstack(
        (np.transpose(cal_R[:, 0:numTrain]), I))
    y = vol_Force[0:numTrain, 4]
    h = (np.transpose(x) * x).I * np.transpose(x) * y.reshape(-1, 1)
    # print('Mz = ', h[0, 0], ' * R13 + ', h[1, 0], ' * R23 + ', h[2, 0], ' * R33 + ',
    #       h[3, 0])

    I = np.mat([1 for _ in range(numTrain, row)]).reshape(-1, 1)
    x_test = np.hstack((np.transpose(cal_R[:, numTrain:row]), I))
    y_test = vol_Force[numTrain:row, 4]
    M_error = y_test - np.dot(x_test, h)
    # print('Mz error= \n', Mz_error)
    length = len(list(M_error))
    plt.scatter([i for i in range(length)], list(M_error))
    # plt.ylim(-10, 10)
    plt.title('Linear My_error')
    plt.show()

    '''Mz'''
    I = np.mat([1 for _ in range(numTrain)]).reshape(-1, 1)
    x = np.hstack(
        (np.transpose(cal_R[:, 0:numTrain]), I))
    y = vol_Force[0:numTrain, 5]
    h = (np.transpose(x) * x).I * np.transpose(x) * y.reshape(-1, 1)
    # print('Mz = ', h[0, 0], ' * R13 + ', h[1, 0], ' * R23 + ', h[2, 0], ' * R33 + ',
    #       h[3, 0])

    I = np.mat([1 for _ in range(numTrain, row)]).reshape(-1, 1)
    x_test = np.hstack((np.transpose(cal_R[:, numTrain:row]), I))
    y_test = vol_Force[numTrain:row, 5]
    M_error = y_test - np.dot(x_test, h)
    # print('Mz error= \n', Mz_error)
    length = len(list(M_error))
    plt.scatter([i for i in range(length)], list(M_error))
    # plt.ylim(-10, 10)
    plt.title('Linear Mz_error')
    plt.show()


'''
    Pytorch是一个拥有强力GPU加速的张量和动态构建网络的库，其主要构建是张量，所以可以把PyTorch当做Numpy
    来用,Pytorch的很多操作好比Numpy都是类似的，但是其能够在GPU上运行，所以有着比Numpy快很多倍的速度。
    训练完了，发现隐层越大，拟合的速度越是快，拟合的效果越是好
'''


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        # 初始网络的内部结构
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, 45)
        self.hidden1 = torch.nn.Linear(45, 90)
        self.hidden2 = torch.nn.Linear(90, 135)
        self.hidden3 = torch.nn.Linear(135, 90)
        self.hidden4 = torch.nn.Linear(90, 45)
        self.predict = torch.nn.Linear(45, n_output)

    def forward(self, x):
        # 一次正向行走过程
        x = F.leaky_relu(self.hidden(x))
        x = F.leaky_relu(self.hidden1(x))
        x = F.leaky_relu(self.hidden2(x))
        x = F.leaky_relu(self.hidden3(x))
        x = F.leaky_relu(self.hidden4(x))
        x = self.predict(x)
        return x


def trainByPytorch(conForce, TCPRv):
    print('------      构建数据集      ------')
    conForceList = np.transpose(conForce).tolist()
    TCPRvList = np.transpose(TCPRv).tolist()

    x = torch.unsqueeze(torch.tensor(TCPRvList[0]), dim=1)
    for i in range(1, 9):
        x_temp = torch.unsqueeze(torch.tensor(TCPRvList[i]), dim=1)
        x = torch.cat((x, x_temp), 1)

    y = torch.unsqueeze(torch.tensor(conForceList[0]), dim=1)
    for i in range(1, 6):
        y_temp = torch.unsqueeze(torch.tensor(conForceList[i]), dim=1)
        y = torch.cat((y, y_temp), 1)

    x, y = Variable(x), Variable(y)

    x = x.to(device)
    y = y.to(device)

    print('------      搭建网络      ------')
    # 使用固定的方式继承并重写 init和forword两个类
    net = Net(n_feature=9, n_hidden=100, n_output=6)

    print('网络结构为：', net)

    print('------      启动训练      ------')
    loss_func = F.mse_loss
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net = net.to(device)
    # 使用数据 进行正向训练，并对Variable变量进行反向梯度传播  启动100次训练
    for t in range(100000):
        if t < 10000:
            optimizer = torch.optim.Adam(net.parameters(), lr=0.005,weight_decay=0.001)
        elif t < 30000:
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001,weight_decay=0.001)
        elif t < 50000:
            optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=0.001)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=0.0001,weight_decay=0.001)
        # 使用全量数据 进行正向行走

        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()  # 清除上一梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 应用梯度
        if t % 10 == 0:
            print(t,':',loss.item())
            if (loss.item() < 228):
                torch.save(net.state_dict(), r'.\两层ReLU9800组矩阵到电压.pth')
    print('------      预测和可视化      ------')


def predictError(path, conForce, TCPRv, conForceScaler):
    net2 = Net(n_feature=9, n_hidden=100, n_output=6)
    net2.load_state_dict(torch.load(path))
    net2 = net2.to(device)
    conForceList = np.transpose(conForce).tolist()
    TCPRvList = np.transpose(TCPRv).tolist()
    x = torch.unsqueeze(torch.tensor(TCPRvList[0]), dim=1)
    for i in range(1, 9):
        x_temp = torch.unsqueeze(torch.tensor(TCPRvList[i]), dim=1)
        x = torch.cat((x, x_temp), 1)

    y = torch.unsqueeze(torch.tensor(conForceList[0]), dim=1)
    for i in range(1, 6):
        y_temp = torch.unsqueeze(torch.tensor(conForceList[i]), dim=1)
        y = torch.cat((y, y_temp), 1)

    x, y = Variable(x), Variable(y)
    x = x.to(device)
    y = y.to(device)
    prediction = net2(x)
    y = y.data.cpu().numpy()
    prediction = prediction.data.cpu().numpy()
    # y = conForceScaler.inverse_transform(y)
    prediction = conForceScaler.inverse_transform(prediction)
    k = np.array([[-0.508355458, -0.010221245, 0.001899991, 0.017937002, -0.89303229, -0.078759786],
                  [0.01629597, -0.509286966, 0.003531495, 0.890626257, 0.034062921, -0.107769097],
                  [-0.003018283, 0.002707507, 0.167344678, 0.013296802, 5.05802E-05, 0.003378879],
                  [-0.00482948, 0.670421448, -0.009244749, -2.482150596, -0.020502753, 0.124676258],
                  [-0.678125117, -0.022025622, 0.004429708, 0.080445601, -2.492087999, -0.123587686],
                  [0.000117831, 0.000621971, -0.000255144, -0.000579705, 0.002064402, 0.777330755]])
    V0 = np.array([[1989.26092624, 1856.18455229, 2438.63260211, 1757.63199822, 1718.09624644, 1918.89437554] for _ in
                   range(len(y))])
    prediction = prediction - V0
    prediction = np.dot(prediction, np.transpose(k))
    y = y - V0
    y = np.dot(y, np.transpose(k))
    error = prediction - y
    # error = abs(error)
    errorMax = np.max(error, axis=0)
    errorMean = np.mean(error, axis=0)
    errorStd = np.std(error, axis=0)
    print('Max: ', errorMax)
    print('Mean: ', errorMean)
    print('Std: ', errorStd)
    plt.figure(1)
    plt.subplot(2, 3, 1)
    plt.plot(error[:, 0], color='g', label='Fx')
    plt.title('FxError')
    plt.xlabel('number')
    plt.ylabel('error/N')
    plt.figure(1)
    plt.subplot(2, 3, 2)
    plt.plot(error[:, 1], color='r', label='Fy')
    plt.title('FyError')
    plt.xlabel('number')
    plt.ylabel('error/N')
    plt.figure(1)
    plt.subplot(2, 3, 3)
    plt.plot(error[:, 2], color='y', label='Fz')
    plt.title('FzError')
    plt.xlabel('number')
    plt.ylabel('error/N')
    plt.figure(1)
    plt.subplot(2, 3, 4)
    plt.plot(error[:, 3], color='b', label='Mx')
    plt.title('MxError')
    plt.xlabel('number')
    plt.ylabel('error/Ncm')
    plt.figure(1)
    plt.subplot(2, 3, 5)
    plt.plot(error[:, 4], color='c', label='My')
    plt.title('MyError')
    plt.xlabel('number')
    plt.ylabel('error/Ncm')
    plt.subplot(2, 3, 6)
    plt.plot(error[:, 5], color='m', label='My')
    plt.title('MzError')
    plt.xlabel('number')
    plt.ylabel('error/Ncm')
    plt.show()

def predict(path, conForce, TCPRv):

    net2 = Net(n_feature=3, n_hidden=100, n_output=6)
    net2.load_state_dict(torch.load(path))
    net2 = net2.to(device)
    conForceList = np.transpose(conForce).tolist()
    TCPRvList = np.transpose(TCPRv).tolist()
    x = torch.unsqueeze(torch.tensor(TCPRvList[0]), dim=1)
    for i in range(1, 3):
        x_temp = torch.unsqueeze(torch.tensor(TCPRvList[i]), dim=1)
        x = torch.cat((x, x_temp), 1)

    y = torch.unsqueeze(torch.tensor(conForceList[0]), dim=1)
    for i in range(1, 6):
        y_temp = torch.unsqueeze(torch.tensor(conForceList[i]), dim=1)
        y = torch.cat((y, y_temp), 1)

    x, y = Variable(x), Variable(y)
    x = x.to(device)
    y = y.to(device)
    prediction = net2(x)
    error = prediction-y
    print(prediction.data.cpu().numpy()[0])
    print(error.data.cpu().numpy()[0])

def normalization(X):
    scaler = preprocessing.StandardScaler().fit(X)
    return scaler,scaler.transform(X),scaler.mean_,scaler.var_



if __name__ == '__main__':
    # trainPath = r'D:\VScode\VScodePython\DataBasedUR\20210106\allDataMedian.xls'
    trainPath = r'D:\VScode\VScodePython\DataBasedUR\20210106矩阵到电压实验效果\train.xls'  # xls dir
    testPath = r'D:\VScode\VScodePython\DataBasedUR\20210106矩阵到电压实验效果\test.xls'
    conForce, TCPRv, RPYangle,V = readData(trainPath)
    conForceTest, TCPRv, RPYangleTest,VTest = readData(testPath)
    # traForce, conForce, TCPRv, RPYangle, V = readDataAll(trainPath)
    scaler, scalerV, mean, var = normalization(V)
    '''画出原始数据'''
    # plt.figure(1)
    # plt.subplot(2, 3, 1)
    # plt.plot(V[:, 0], color='g', label='Fx')
    # plt.figure(1)
    # plt.subplot(2, 3, 2)
    # plt.plot(V[:, 1], color='r', label='Fy')
    # plt.figure(1)
    # plt.subplot(2, 3, 3)
    # plt.plot(V[:, 2], color='y', label='Fz')
    # plt.figure(1)
    # plt.subplot(2, 3, 4)
    # plt.plot(V[:, 3], color='b', label='Mx')
    # plt.figure(1)
    # plt.subplot(2, 3, 5)
    # plt.plot(V[:, 4], color='c', label='My')
    # plt.figure(1)
    # plt.subplot(2, 3, 6)
    # plt.plot(V[:, 5], color='m', label='Mz')
    # plt.show()

    '''计算旋转矩阵，并取其转置的最后一列'''
    RM = RV2RM(TCPRv)
    # RM13 = []
    # RM23 = []
    # RM33 = []
    # for i in range(TCPRv.shape[0]):  #末端到基坐标旋转矩阵的转置后的最后一列
    #     RM13.append(RM[i][2,0])
    #     RM23.append(RM[i][2,1])
    #     RM33.append(RM[i][2,2])
    #
    # RM3 = [RM13,RM23,RM33]
    # RM3 = np.array(RM3)  #3*4732
    # RM3 = np.transpose(RM3) #转为4732*3
    #
    # plt.figure(2)
    # plt.plot(RM33,color = 'red')
    # plt.show()
    # '''取出每个矩阵的9个元素'''
    input = []
    for i in range(len(RM)):
        arrayList = []
        for m in range(3):
            for n in range(3):
                arrayList.append(RM[i][m,n])
        input.append(arrayList)
    input = np.array(input)
    '''训练'''
    # trainByPytorch(scalerV ,input) #改写输入为旋转矩阵的最后一列
    predictError("四层9800组矩阵到电压.pth", VTest , input,scaler)
