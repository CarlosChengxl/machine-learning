import numpy as np
import matplotlib.pyplot as plt
from pip import main
import scipy.io as sio


def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma2 = np.sum(np.power(X - mu, 2), 0) / X.shape[0]

    return mu, sigma2


def gaussian_distribution(X, mu, sigma2):
    p = (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-np.power((X - mu), 2) / (2 * sigma2))
    return np.prod(p, axis=1)           # 横向连乘


def select_threshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    for epsilon in np.linspace(np.min(pval), np.max(pval), num=1001):

        predictions = np.where(pval < epsilon, 1, 0)  # 小于epsilon则预测为异常（为真）， 满足条件返回1，不满足返回0
        tp = np.sum(yval[np.where(predictions == 1)])  # 正确肯定，上面预测的为1，并且yval中也是1
        fp = np.sum(np.where(yval[np.where(predictions == 1)] == 0, 1, 0))  # 错误肯定，预测为真，实际为假
        fn = np.sum(np.where(yval[np.where(predictions == 0)] == 1, 1, 0))  # 错误否定

        if tp + fp == 0 or tp + fn == 0:
            continue

        precision = tp / (tp + fp)  # 查准率
        recall = tp / (tp + fn)  # 查全率
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1


def detection(X, eps, mu, sigma2):
    px = gaussian_distribution(X, mu, sigma2)
    # 进行判断异常点
    anomaly_points = X[np.where(px < eps)]
    return anomaly_points


def visualize_countors(mu, sigma2):  # 绘制高斯分布等高线
    # 由X分布可知，选取5-25，范围比较合适
    X = np.linspace(5, 25, 100)
    y = np.linspace(5, 25.100)

    xx, yy = np.meshgrid(X, y)
    X = np.c_[xx.flatten(), yy.flatten()]  # 数据对应网格中每一个点
    z = gaussian_distribution(X, mu, sigma2).reshape(xx.shape)  # 获取每一个点坐标的高斯值
    cont_levels = [10 ** h for h in range(-20, 0, 3)]  # 当z为当前列表的值才绘出等高线（最高1） 不设置的话，会比较奇怪

    plt.contour(xx, yy, z, cont_levels)


def circle_anomaly_points(X):
    plt.scatter(X[:, 0], X[:, 1], s=80, facecolors='none', edgecolors='r')


if __name__ == '__main__':
    data = sio.loadmat("./data/ex8data2.mat")
    X = data['X']
    print(X.shape)
    Xval = data['Xval']
    yval = data['yval']

    mu, sigma2 = estimate_gaussian(X)
    px = gaussian_distribution(Xval, mu, sigma2)
    esp, f1 = select_threshold(px, yval)
    print(esp, f1)

    anomaly_points = detection(X, esp, mu, sigma2)
    print(anomaly_points.shape)
