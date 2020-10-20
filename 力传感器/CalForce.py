import os
import xlrd
import xlwt
import numpy as np
import string

'''
读取零点
'''
zeroPath = r'D:\1A.研究生\基于力的位姿计算\20200928采集\0\Avg.xls'
zeroData = xlrd.open_workbook(zeroPath).sheets()[0]
nrows = zeroData.nrows
zeroList = []
for i in range(nrows):
    zeroList.append(zeroData.row_values(i)[1:7])
zeroMat = np.mat(zeroList)

path = r'D:\1A.研究生\基于力的位姿计算\20200928采集'

for folders in os.listdir(path):
    xlsPath = os.path.join(path, folders, 'Avg.xls')
    savePath = os.path.join(path, folders, 'Force.xls')
    data = xlrd.open_workbook(xlsPath).sheets()[0]
    nrows = data.nrows
    dataList = []
    for i in range(nrows):
        dataList.append(data.row_values(i)[1:7])
    dataMat = np.mat(dataList)
    forceMat = dataMat-zeroMat

    Book = xlwt.Workbook(encoding='ascii', style_compression=0)
    Sheet = Book.add_sheet('ForceAvg', cell_overwrite_ok=True)
    m, n = np.shape(forceMat)
    for i in range(m):
        for j in range(n):
            Sheet.write(i,   j, forceMat[i,j])
    Book.save(savePath)
