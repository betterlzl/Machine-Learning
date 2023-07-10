import numpy as np
import pandas as pd
import pylab as plt

#第1步：加载数据集
dataset = pd.read_csv("Al_CEEMDANR.csv", parse_dates=['date'], index_col=['date'])
#data.drop(labels='Unnamed: 0', axis=1, inplace=True)
#print("max close:",data.close.max() )


##++++++++ 定义样本熵SE +++++++++++
def SampEn(U, m, r):
    """
    用于量化时间序列的可预测性
    :param U: 时间序列
    :param m: 模板向量维数
    :param r: 距离容忍度，一般取0.1~0.25倍的时间序列标准差，也可以理解为相似度的度量阈值
    :return: 返回一个-np.log(A/B)，该值越大，序列就越复杂
    """

    def _maxdist(x_i, x_j):
        """
         Chebyshev distance
        :param x_i:
        :param x_j:
        :return:
        """
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
        result = sum(C) / (N - m)
        return result

    N = len(U)
    return -np.log(_phi(m + 1) / _phi(m))

##++++++ 计算样本熵SE +++++++
if __name__ == '__main__':
    m = 2
    Data1 = dataset['A']
    print('Data1\n',Data1)
    SE1 = SampEn(Data1, m, r=0.2 * np.std(Data1))
    print("SE:", SE1)

    Data2 = dataset['B']
    SE2 = SampEn(Data2, m, r=0.2 * np.std(Data2))
    print("SE:", SE2)

    Data3 = dataset['C']
    SE3 = SampEn(Data3, m, r=0.2 * np.std(Data3))
    print("SE:", SE3)

    Data4 = dataset['D']
    SE4 = SampEn(Data4, m, r=0.2 * np.std(Data4))
    print("SE:", SE4)

    Data5 = dataset['E']
    SE5 = SampEn(Data5, m, r=0.2 * np.std(Data5))
    print("SE:", SE5)

    Data6 = dataset['F']
    SE6 = SampEn(Data6, m, r=0.2 * np.std(Data6))
    print("SE:", SE6)


    Data7 = dataset['G']
    SE7 = SampEn(Data7, m, r=0.2 * np.std(Data7))
    print("SE:", SE7)

    Data8 = dataset['H']
    SE8 = SampEn(Data8, m, r=0.2 * np.std(Data8))
    print("SE:", SE8)

    Data9 = dataset['I']
    SE9 = SampEn(Data9, m, r=0.2 * np.std(Data9))
    print("SE:", SE9)

    SE = [SE1,SE2,SE3,SE4,SE5,SE6,SE7,SE8,SE9]
    # 绘制样本熵曲线
    x0 = range(1, 10, 1)
    plt.figure(figsize=(12,6) )
    plt.plot(x0, SE, color='mediumpurple', linestyle='--', marker='o', markerfacecolor='w' )
    plt.xlabel('IMFs', fontsize='20')
    plt.ylabel("Sample entropy", fontsize='20')
    # 刻度值字体大小设置（x轴和y轴同时设置）
    plt.tick_params(labelsize=15)
    plt.savefig('./aSE.png')
    plt.savefig('./aSE.eps')
    plt.show()


