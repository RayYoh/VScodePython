import xlrd
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



""" Read experimental data """
# For UR robot
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
    volList = []
    for i in range(0, Sheet_rows):
        dataLine = Sheet.row_values(i)
        rawTraForce = [f for f in dataLine[15:21]]
        rawConForce = [f for f in dataLine[35:41]]
        rawTCPPos = [f for f in dataLine[42:48]]
        rawRPY = util.rv2rpy(rawTCPPos[3], rawTCPPos[4], rawTCPPos[5])  # Rotation Vector 2 RPY
        volList.append(rawTraForce+rawConForce+rawTCPPos+list(rawRPY))
    vol = np.mat(volList)
    # np.random.shuffle(vol)
    traForce = vol[:, :6]
    conForce = vol[:, 6:12]
    TCPRv = vol[:, 15:18]
    RPYangle = vol[:, 18:]

    return traForce, conForce,TCPRv, RPYangle

def RPY2RM(RPYangle):
    '''Calculate rotation matrix from RPY angle
    Args:
        RPYangle: RPY angle  type；matrix

    Return:
        RM: Rotation Matrix type:array
    '''
    row = np.shape(RPYangle)[0]
    RM = []
    for i in range(row):
        RM.append(util.rpy2rm(RPYangle[i].A[0]))  # Change matrix to array, and take the first element.
    RM = np.array(RM)
    return  RM


"""Calculate the transpose rotation matrix"""
def CalRatMat(RPY_angle):
    # For PUMA Robot
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
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden1 = torch.nn.Linear(n_hidden, 500)
        self.hidden2 = torch.nn.Linear(500, 100)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # 一次正向行走过程
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x


def trainByPytorch(conForce, TCPRv):
    print('------      构建数据集      ------')
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

    print('------      搭建网络      ------')
    # 使用固定的方式继承并重写 init和forword两个类
    net = Net(n_feature=3, n_hidden=100, n_output=6)

    print('网络结构为：', net)

    print('------      启动训练      ------')
    loss_func = F.mse_loss
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net = net.to(device)
    # 使用数据 进行正向训练，并对Variable变量进行反向梯度传播  启动100次训练
    for t in range(5000):
        # if t < 5000:
        #     optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        # else:
        #     optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
        # 使用全量数据 进行正向行走

        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()  # 清除上一梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 应用梯度

        # 间隔一段，对训练过程进行可视化展示
        '''
        if t % 5 == 0:
            plt.cla()
            plt.scatter(x.data.cpu().numpy(), y.data.cpu().numpy())  # 绘制真实曲线

            x_gt = x.data.cpu().numpy().copy()
            y_pre = prediction.data.cpu().numpy().copy()
            x_gt = np.transpose(x_gt)
            y_pre = np.transpose(y_pre)
            sorted_indx = x_gt.argsort() #x增序后的索引
            y_pre_backup = y_pre.copy()
            n = 0
            for i in sorted_indx:
                for j in i:
                    y_pre[0][n] = y_pre_backup[0][j]
                    n += 1

            x_gt = x_gt[0]
            y_pre = y_pre[0]
            x_gt.sort()

            plt.scatter(x_gt, y_pre)
            plt.plot(x_gt, y_pre, c='r')
            plt.text(0.5, 0, 'Loss='+str(loss.item()),
                     fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)
            if(loss.item()<30.58):
                torch.save(net.state_dict(), '.\Mx.pth')
        '''
        if t % 10 == 0:
            print(loss.item())
            if (loss.item() < 2.91):
                torch.save(net.state_dict(), '.\RPY2Force.pth')
    # plt.ioff()
    # plt.show()
    print('------      预测和可视化      ------')


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
    # print(error.data.cpu().numpy()[:,0])
    plt.scatter([i for i in range(len(error.data.cpu().numpy()))],
                error.data.cpu().numpy()[:,4])
    # plt.ylim(-10,10)
    plt.title('ForceError')
    plt.show()



if __name__ == '__main__':
    data_path = r'D:\VScode\VScodePython\DataBasedUR\Data\allData.xls'  # xls dir
    traForce, conForce, TCPRv, RPYangle = readData(data_path) # Use the m & rad
    RM = RPY2RM(RPYangle)

    # plt.scatter(list(cal_R[0,:]),list(vol_Force[:,0]))
    # plt.xlabel('R13')
    # plt.ylabel('Fx')
    # plt.show()

    # trainByPytorch(conForce,RPYangle)
    predict("RPY2Force.pth", conForce, RPYangle)

    # train_Lin(vol_Force, cal_R, 15000)
