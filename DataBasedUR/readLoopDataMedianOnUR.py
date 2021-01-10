import os
import xlwt
import xlrd
import util
import numpy as np

def read():
    dataDir = r'D:\VScode\VScodePython\DataBasedUR\20210106'
    saveDir = dataDir + r"\allDataMedian.xls"

    dataList = os.listdir(dataDir)
    # dataList.sort()


    if 'allDataMedian.xls' in dataList:
        dataList.remove('allDataMedian.xls')
    pointNum = 0
    allData = []
    for dataTxt in dataList:
        pointNum += 1
        oneData = []
        with open(dataDir+'/'+dataTxt, "r") as f:
            collectNum = 0
            for line in f:
                collectNum += 1
                line = line.strip('\n')
                line = line.split(',')
                newLine = line[1:13] + line[14:26] + line[27:33] + line[34:40] + \
                    line[41:47] + line[48:54] + line[55:61] + line[62:68]
                newLine = [float(x) for x in newLine]
                oneData.append(newLine)
        oneData = np.array(oneData)
        saveData = np.median(oneData, axis=0)
        saveData.tolist()
        allData.append(saveData)


    allData = np.array(allData, dtype=object )
    row= allData.shape[0]
    allName = ['No.', 'Traction sensor V', 'Traction sensor F/M', 'Contact sensor V', 'Contact sensor F/M',
               'Robot TCP position', 'Robot Joint position', 'Robot TCP Torque', 'Robot Joint Torque']
    # Save
    Book = xlwt.Workbook(encoding='ascii', style_compression=0)
    Sheet = Book.add_sheet('AllData', cell_overwrite_ok=True)

    for i in range(row):
        Sheet.write(i, 0, allName[0]+str(i+1))
        Sheet.write(i, 1, allName[1])
        Sheet.write(i, 14, allName[2])
        Sheet.write(i, 27, allName[3])
        Sheet.write(i, 34, allName[4])
        Sheet.write(i, 41, allName[5])
        Sheet.write(i, 48, allName[6])
        Sheet.write(i, 55, allName[7])
        Sheet.write(i, 62, allName[8])
        for j in range(12):
            Sheet.write(i,j+2,allData[i,j])
            Sheet.write(i,j+15,allData[i,j+12])
        for j in range(6):
            Sheet.write(i,j+28,allData[i,j+24])
            Sheet.write(i,j+35,allData[i,j+30])
            Sheet.write(i,j+42,allData[i,j+36])
            Sheet.write(i,j+49,allData[i,j+42])
            Sheet.write(i,j+56,allData[i,j+48])
            Sheet.write(i,j+63,allData[i,j+54])
    Book.save(saveDir)
    print('Finished! There are %d in data set!' % pointNum)

def saveGroup():
    path = r'D:\VScode\VScodePython\DataBasedUR\20210106\allDataMedian.xls'
    trainPath = r'D:\VScode\VScodePython\DataBasedUR\20210106矩阵到电压实验效果\train.xls'
    testPath = r'D:\VScode\VScodePython\DataBasedUR\20210106矩阵到电压实验效果\test.xls'
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
    np.random.shuffle(vol)
    traForce = vol[:, :6]
    conForce = vol[:, 6:12]
    TCPRv = vol[:, 15:18]
    RPYangle = vol[:, 18:]
    V = vol[:, 21:]
    # Save Train
    Book = xlwt.Workbook(encoding='ascii', style_compression=0)
    Sheet = Book.add_sheet('AllData', cell_overwrite_ok=True)
    for i in range (9800):
        for j in range(6):
            Sheet.write(i,j,conForce[i,j])
            Sheet.write(i, j+6, V[i, j])
        for j in range(3):
            Sheet.write(i,j+12,TCPRv[i,j])
    Book.save(trainPath)
    print("Finish Train Data!")
    # Save Train
    Book = xlwt.Workbook(encoding='ascii', style_compression=0)
    Sheet = Book.add_sheet('AllData', cell_overwrite_ok=True)
    for i in range(9800,Sheet_rows):
        for j in range(6):
            Sheet.write(i-9800, j, conForce[i, j])
            Sheet.write(i-9800, j + 6, V[i, j])
        for j in range(3):
            Sheet.write(i-9800, j + 12, TCPRv[i, j])
    Book.save(testPath)
    print("Finish Test Data!")

if __name__ == '__main__':
    saveGroup()