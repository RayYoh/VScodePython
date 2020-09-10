import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

'''
海伦约会
'''

def file2mat(filename):
    '''
    打开文件解析数据
    :param filename:文件名
    :return dataMat:数据矩阵 labels：标签向量
    '''
    fr = open(filename)
    #全部读取
    arrayOfLines = fr.readlines()
    #数据行数
    numOfLines = len(arrayOfLines)
    dataMat = np.zeros((numOfLines, 3))
    labels = []
    index = 0
    for line in arrayOfLines:
        #取出首尾空格
        line = line.strip()
        #将数据按\t切片
        listFromLine = line.split('\t')
        #读取
        dataMat[index, :] = listFromLine[0:3]
        '''
        1:didntLike 2:smallDoses 3:largeDoses
        '''
        if listFromLine[-1] == 'didntLike':
            labels.append(1)
        elif listFromLine[-1] == 'smallDoses':
            labels.append(2)
        elif listFromLine[-1] == 'largeDoses':
            labels.append(3)
        index += 1
    return dataMat, labels


def showDatas(dataMat, labels):
    '''
    显示数据
    :param dataMat:读取的数据 
    :param labels:标签向量
    '''
    #图像2行2列共四个区域，不共享xy轴
    fig, axs = plt.subplots(nrows=2,
                            ncols=2,
                            sharex=False,
                            sharey=False,
                            figsize=(13, 8))
    numOfLabels = len(labels)
    labelsColors = []
    #颜色分类
    for i in labels:
        if i == 1:
            labelsColors.append('green')
        elif i == 2:
            labelsColors.append('blue')
        elif i == 3:
            labelsColors.append('red')

    #设置第一个区域xy轴数据，颜色，点的大小透明度
    axs[0][0].scatter(x=dataMat[:, 0],
                      y=dataMat[:, 1],
                      color=labelsColors,
                      s=15,
                      alpha=0.5)
    #图像标题
    axs0TitleText = axs[0][0].set_title('flyMiles and gameTime')
    #图像x，y轴
    axs0XlabelText = axs[0][0].set_xlabel('flyMiles')
    axs0YlabelText = axs[0][0].set_ylabel('gameTime')
    #图像标题大小颜色
    plt.setp(axs0TitleText, size=12, weight='bold', color='red')
    #图像x，y轴大小颜色
    plt.setp(axs0XlabelText, size=9, weight='bold', color='black')
    plt.setp(axs0YlabelText, size=9, weight='bold', color='black')

    #第二个区域
    axs[0][1].scatter(x=dataMat[:, 0],
                      y=dataMat[:, 2],
                      color=labelsColors,
                      s=15,
                      alpha=0.5)
    axs1TitleText = axs[0][1].set_title('flyMiles and iceCream')
    axs1XlabelText = axs[0][1].set_xlabel('flyMiles')
    axs1YlabelText = axs[0][1].set_ylabel('iceCream')
    plt.setp(axs1TitleText, size=12, weight='bold', color='red')
    plt.setp(axs1XlabelText, size=9, weight='bold', color='black')
    plt.setp(axs1YlabelText, size=9, weight='bold', color='black')

    #第三个区域
    axs[1][0].scatter(x=dataMat[:, 1],
                      y=dataMat[:, 2],
                      color=labelsColors,
                      s=15,
                      alpha=0.5)
    axs2TitleText = axs[1][0].set_title('gameTime and iceCream')
    axs2XlabelText = axs[1][0].set_xlabel('gameTime')
    axs2YlabelText = axs[1][0].set_ylabel('iceCream')
    plt.setp(axs2TitleText, size=12, weight='bold', color='red')
    plt.setp(axs2XlabelText, size=9, weight='bold', color='black')
    plt.setp(axs2YlabelText, size=9, weight='bold', color='black')

    #设置图例
    didntLike = mlines.Line2D([], [],
                              color='green',
                              marker='.',
                              markersize=6,
                              label='didntLike')
    smallDoses = mlines.Line2D([], [],
                               color='blue',
                               marker='.',
                               markersize=6,
                               label='smallDoses')
    largeDoses = mlines.Line2D([], [],
                               color='red',
                               marker='.',
                               markersize=6,
                               label='largeDoses')

    #添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])

    plt.show()


def autoNorm(dataMat):
    '''
    归一化数据
    :param dataMat:读取的数据
    :return normMat:归一化后的数据 minVals:最小值 maxVals:最大值
    '''
    #按列计算最小值与最大值
    minVals = dataMat.min(axis=0)
    maxVals = dataMat.max(axis=0)
    # normMat=np.zeros(dataMat.shape)
    m = dataMat.shape[0]
    #归一化：normMat=(dataM-minVals)/(maxVals-minVals)
    normMat = (dataMat - np.tile(minVals,
                                 (m, 1))) / np.tile(maxVals - minVals, (m, 1))
    return normMat, minVals, maxVals


def datingClassTest(dataMat, labels):
    '''
    分类器测试函数
    :param dataMat:读取的数据
    :param labels:标签向量
    '''
    #测试集所占的比例
    testRatio = 0.10
    dataMat, minVals, maxVals = autoNorm(dataMat)
    m = dataMat.shape[0]
    #测试集个数
    numTest = int(m * testRatio)
    #生成模型
    model = kNN(dataMat[numTest:m], labels[numTest:m])
    #误分类个数
    errorCount = 0.0
    for i in range(numTest):
        classfyResult = model.classfy(dataMat[i, :], 3)
        print('分类结果：%d 真实结果：%d' % (classfyResult, labels[i]))
        if classfyResult != labels[i]:
            errorCount += 1.0
    print('错误率：%f%%' % (errorCount / float(numTest) * 100))


def classfyPerson(dataMat, labels):
    '''
    分类函数
    :param dataMat:读取的数据
    :param labels:标签向量
    '''
    resultList = ['didntLike', 'smallDoses', 'largeDoses']
    dataMat, minVals, maxVals = autoNorm(dataMat)
    #输入待分类信息
    flyMiles = float(input('每年获得的飞行常客里程数：'))
    gameTime = float(input('玩视频游戏所耗时间百分比：'))
    iceCream = float(input('每周消费的冰淇淋公升数：'))
    inArr = np.array([flyMiles, gameTime, iceCream])
    #输入数据归一化
    normInArr = inArr - minVals / (maxVals - minVals)
    model = kNN(dataMat, labels)
    result = model.classfy(normInArr, 3)
    print('你可能%s这个人' % resultList[result - 1])


class kNN(object):
    '''
    k近邻算法
    '''
    def __init__(self, dataMat, labels):
        '''
        初始化模型数据
        ：param dataMat：训练数据集
        ：param labels：训练数据集的标签
        '''
        self.dataMat = dataMat
        self.labels = labels

    def classfy(self, inX, k):
        '''
        线性扫描方式分类器
        ：param inX：待分类特征向量
        ：param k：k值
        '''
        inX = np.tile(
            inX, (self.dataMat.shape[0], 1))  #np.tile(x,y)表示将inX行重复x次列重复y次
        '''
        计算欧式距离:
        '''
        distance = np.sum((inX - self.dataMat)**2, axis=1)**0.5
        sortDistanceIndex = np.argsort(distance)  #对distance进行排序，并获得排序后的索引
        labelCount = {}  #创建字典，用于存储标签个数
        for i in range(k):
            selectLabel = self.labels[
                sortDistanceIndex[i]]  #利用循环从k个近邻中选取从最近邻开始到k的标签
            labelCount[selectLabel] = labelCount.get(
                selectLabel, 0) + 1  #存储标签个数，字典的get(*,0)函数，若为None则返回0

        sortLabelCount = sorted(labelCount.items(),
                                key=lambda item: item[1],
                                reverse=True)  #根据字典的value值排序，多数原则
        return sortLabelCount[0][0]  #返回最大个数的标签


if __name__ == '__main__':
    filename = 'D:\VScodePython\机器学习算法\k近邻\datingTestSet.txt'
    data, labels = file2mat(filename)
    showDatas(data, labels)
    #datingClassTest(data, labels)
    #classfyPerson(data, labels)