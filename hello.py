import numpy as np  ##科学计算库
import scipy as sp  ##在numpy基础上实现的部分算法库
import matplotlib.pyplot as plt  ##绘图库
from scipy.optimize import leastsq  ##引入最小二乘法算法

'''
x = np.mat([[-0.99998629, 1], [-0.38912395, 1], [-0.46329604, 1],
            [-0.74431155, 1], [-0.80798988, 1], [-0.93295353, 1]])
y = np.array([-44.841, 5.945, -18.068, -29.269, -35.271, -25.767])
print("x:\n", x)
print("y:\n", y)
h = (np.transpose(x) * x).I * np.transpose(x) * y.reshape(-1, 1)
print(h)
'''

x=np.array([-0.98684219 , 0.12089164, -0.41954643 ,-0.68564893, -0.79782037 ,-0.90484552])
y = np.array([5.923,26.325,-14.055,-24.401, -9.681,-47.831])
def func(p, x):
    k, b = p
    return k * x + b 


def error(p, x, y):
    return func(p, x) - y


p0 = [1, 0]
#把error函数中除了p0以外的参数打包到args中(使用要求)
Para = leastsq(error, p0, args=(x, y))

#读取结果
k, b = Para[0]
print("k=", k, "b=", b)
print("cost：" + str(Para[1]))
print("求解的拟合直线为:")
print("y=" + str(round(k, 2)) + "x+" + str(round(b, 2)))


#画样本点
plt.figure(figsize=(8, 6))  ##指定图像比例： 8：6
plt.scatter(x, y, color="green", label="样本数据", linewidth=2)

#画拟合直线
x = np.linspace(-1, 0.2, 100)  ##在0-15直接画100个连续点
y = k * x+ b   ##函数式
plt.plot(x, y, color="red", label="拟合直线", linewidth=2)
plt.legend(loc='lower right')  #绘制图例
plt.show()
