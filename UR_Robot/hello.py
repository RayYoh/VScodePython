import numpy as np

allData = []
for i in range(3):
    oneData = [[1,2,3],[4,5,6]]
    oneData = np.array(oneData)
    saveData = np.median(oneData,axis=0)
    saveData.tolist()
    print(saveData)
    allData.append(saveData)
allData = np.array(allData)
print(allData)