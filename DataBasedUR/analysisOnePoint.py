import os
import numpy as np
import matplotlib.pyplot as plt

dataDir = r'D:\VScode\VScodePython\DataBasedUR\20210106'

allData = []
with open(dataDir+'/3.98,4.69,-15.30.txt', "r") as f:
    collectNum = 0
    for line in f:
        collectNum += 1
        line = line.strip('\n')
        line = line.split(',')
        newLine = line[1:13] + line[14:26] + line[27:33] + line[34:40] + \
            line[41:47] + line[48:54] + line[55:61] + line[62:68]
        newLine = [float(x) for x in newLine]
        allData.append(newLine)

conForce = []
for i in range(len(allData)):
    conForce.append(allData[i][30:36])
conForce = np.array(conForce)
conForceMean = []
conForceMedian = []
for i in range(6):
    conForceMean.append(np.mean(conForce[:,i]))
    conForceMedian.append(np.median(conForce[:,i]))
print(conForceMean)
print(conForceMedian)
# conForceFx = list(conForce[:,4])
# minFx = min(conForceFx)
# maxFx = max(conForceFx)
# a = (maxFx - minFx)/5
# bins = [minFx,minFx+a,minFx+2*a,minFx+3*a,minFx+4*a,minFx+5*a]
# b = plt.hist(conForceFx,bins)
# print(b)
# plt.show()
plt.figure(1)
plt.subplot(2, 3, 1)
plt.plot(conForce[:,0],color='g',label = 'Fx')
plt.plot([conForceMedian[0] for _ in range(100)],color='r',label = 'Fx')
plt.figure(1)
plt.subplot(2, 3, 2)
plt.plot(conForce[:,1],color='r',label = 'Fy')
plt.plot([conForceMedian[1] for _ in range(100)],color='r',label = 'Fx')
plt.figure(1)
plt.subplot(2, 3, 3)
plt.plot(conForce[:,2],color='y',label = 'Fz')
plt.plot([conForceMedian[2] for _ in range(100)],color='r',label = 'Fx')
plt.figure(1)
plt.subplot(2, 3, 4)
plt.plot(conForce[:,3],color='b',label = 'Mx')
plt.plot([conForceMedian[3] for _ in range(100)],color='r',label = 'Fx')
plt.figure(1)
plt.subplot(2, 3, 5)
plt.plot(conForce[:,4],color='c',label = 'My')
plt.plot([conForceMedian[4] for _ in range(100)],color='r',label = 'Fx')
plt.figure(1)
plt.subplot(2, 3, 6)
plt.plot(conForce[:,5],color='m',label = 'Mz')
plt.plot([conForceMedian[5] for _ in range(100)],color='r',label = 'Fx')
plt.show()
