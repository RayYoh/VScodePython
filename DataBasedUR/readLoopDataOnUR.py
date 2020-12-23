import os
import xlrd
import xlwt
import numpy as np

dataDir = r'D:\VScode\VScodePython\DataBasedUR\Data'
saveDir = dataDir + r"\allData.xls"

dataList = os.listdir(dataDir)
dataList.sort()


if 'allData.xls' in dataList:
    dataList.remove('allData.xls')
pointNum = 0
allData = []
for dataTxt in dataList:
    pointNum += 1
    sumData = np.zeros(60)
    with open(dataDir+'/'+dataTxt, "r") as f:
        collectNum = 0
        for line in f:
            collectNum += 1
            line = line.strip('\n')
            line = line.split(',')
            newLine = line[1:13] + line[14:26] + line[27:33] + line[34:40] + \
                line[41:47] + line[48:54] + line[55:61] + line[62:68]
            newLine = [float(x) for x in newLine]
            sumData += newLine
    allData.append(sumData/collectNum)


allData = np.array(allData)
row, col = np.shape(allData)
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
for i in range(row):
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