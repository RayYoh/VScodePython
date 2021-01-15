import xlrd
import numpy as np
import util
import RV2V
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

if __name__=='__main__':
    path = r'D:\VScode\VScodePython\DataBasedUR\20210114\吸盘\allDataMedian.xls'
    traForce,conForce,TCPRv,RPYangle,V = readDataAll(path)
    # for i in range(len(conForce)):
    #     print((conForce[i,0]**2+conForce[i,1]**2+conForce[i,2]**2)**0.5)
    Rsb = RV2V.RV2RM(TCPRv)
    R = np.transpose(Rsb[0])
    for i in range(1, len(Rsb)):
        R = np.vstack((R, np.transpose(Rsb[i])))
    FT = np.transpose(conForce)[0:3, :]
    MT = np.transpose(conForce)[3:, :]
    P = FT[:, 0:1]
    Q = MT[:, 0:1]
    for i in range(1, len(TCPRv)):
        P = np.vstack((P, FT[:, i:i + 1]))
        Q = np.vstack((Q, MT[:, i:i + 1]))
    RT = np.transpose(R)
    Mat_R = np.dot(RT, R)
    Mat_R_I = np.linalg.inv(Mat_R)
    h = np.dot(np.dot(Mat_R_I, RT), P)
    print('重力：\n', h)

    H = np.zeros((3 * len(TCPRv), 3))
    for i in range(len(TCPRv)):
        Gx = P[3 * i]
        Gy = P[3 * i + 1]
        Gz = P[3 * i + 2]
        H[3 * i, 1] = Gz
        H[3 * i, 2] = -Gy
        H[3 * i + 1, 0] = -Gz
        H[3 * i + 1, 2] = Gx
        H[3 * i + 2, 0] = Gy
        H[3 * i + 2, 1] = -Gx
    HT = np.transpose(H)
    MatH = np.dot(HT, H)
    MatHI = np.linalg.inv(MatH)
    l = np.dot(np.dot(MatHI, HT), Q)
    print('重心：\n',l)
    #
    # testF = np.dot(R, h)
    # print(P - testF)