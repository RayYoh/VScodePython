#http://sofasofa.io/tutorials/gmm_em/
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')


class GMM(object):
    def __init__(self, X, Mu, Var, Pi):
        '''
        初始化数据
        '''
        self.X = X
        self.Mu = Mu
        self.Var = Var
        self.Pi = Pi
        self.n_clusters, self.n_points = len(Pi), len(X)

    def update_W(self):
        '''
        E步，更新W
        '''
        pdfs = np.zeros((self.n_points, self.n_clusters))
        for i in range(self.n_clusters):
            pdfs[:, i] = Pi[i] * \
                multivariate_normal.pdf(
                    self.X, self.Mu[i], np.diag(self.Var[i]))
        W = pdfs/pdfs.sum(axis=1).reshape(-1, 1)
        return W

    def update_Pi(self, W):
        '''
        E步，更新Pi
        '''
        self.Pi = W.sum(axis=0)/W.sum()
        return self.Pi

    def update_Mu(self, W):
        '''
        M步，更新均值Mu
        '''
        self.Mu = np.zeros((self.n_clusters, 2))
        for i in range(self.n_clusters):
            self.Mu[i] = np.average(self.X, axis=0, weights=W[:, i])
        return self.Mu

    def update_Var(self, W):
        '''
        M步，更新均值Var
        '''
        self.Var = np.zeros((self.n_clusters, 2))
        for i in range(self.n_clusters):
            self.Var[i] = np.average(
                (self.X-self.Mu[i])**2, axis=0, weights=W[:, i])
        return self.Var

    def logLH(self):
        '''
        计算对数似然函数
        '''
        pdfs = np.zeros((self.n_points, self.n_clusters))
        for i in range(self.n_clusters):
            pdfs[:, i] = self.Pi[i] * \
                multivariate_normal.pdf(
                    self.X, self.Mu[i], np.diag(self.Var[i]))
        return np.mean(np.log(pdfs.sum(axis=1)))


def generate_X(true_Mu, true_Var):
'''
生成实验数据
'''
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


def plot_clusters(X, Mu, Var, Mu_true=None, Var_true=None):
'''
画出每一簇数据
'''
    colors = ['b', 'g', 'r']
    n_clusters = len(Mu)
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X[:, 0], X[:, 1], s=5)
    ax = plt.gca()
    for i in range(n_clusters):
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': ':'}
        ellipse = Ellipse(Mu[i], 3 * Var[i][0], 3 * Var[i][1], **plot_args)
        ax.add_patch(ellipse)
    if (Mu_true is not None) & (Var_true is not None):
        for i in range(n_clusters):
            plot_args = {'fc': 'None', 'lw': 2,
                         'edgecolor': colors[i], 'alpha': 0.5}
            ellipse = Ellipse(Mu_true[i], 3 * Var_true[i]
                              [0], 3 * Var_true[i][1], **plot_args)
            ax.add_patch(ellipse)
    plt.show()


'''
模型参数初始化
n_clusters = 3  # GMM中的聚类个数，需提前确定http://sofasofa.io/forum_main_post.php?postid=1001676
n_points = len(X)  # 样本点个数
Mu = [[0, -1], [6, 0], [0, 9]]  # 每个高斯分布均值
Var = [[1, 1], [1, 1], [1, 1]]  # 每个高斯分布的方差，为了过程简便，我们这里假设协方差矩阵都是对角阵
Pi = [1 / n_clusters] * 3  # 每一簇比重 Pi=[1/3,1/3,1/3]
W = np.ones((n_points, n_clusters)) / n_clusters  # 每个样本点属于每一簇的概率
Pi = W.sum(axis=0) / W.sum()
'''

if __name__ == '__main__':
'''
主程序
'''
    #真实生成数据的均值和方差
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)
    
    #初始化
    n_clusters = 3 #簇数
    n_points = len(X) #样本点数
    #迭代初始均值和方差
    Mu = [[0, -1], [6, 0], [0, 9]] 
    Var = [[1, 1], [1, 1], [1, 1]]
    #Pi 是每一簇比重
    Pi = [1 / n_clusters] * 3
    #W 隐变量，每个样本属于每一簇的概率
    W = np.ones((n_points, n_clusters)) / n_clusters
    Pi = W.sum(axis=0) / W.sum()
    # 迭代
    loglh = []
    for i in range(5):
        gmm = GMM(X, Mu, Var, Pi)
        plot_clusters(gmm.X, gmm.Mu, gmm.Var, true_Mu, true_Var)
        loglh.append(gmm.logLH())
        W = gmm.update_W()
        Pi = gmm.update_Pi(W)
        Mu = gmm.update_Mu(W)
        print('log-likehood:%.3f' % loglh[-1])
        Var = gmm.update_Var(W)