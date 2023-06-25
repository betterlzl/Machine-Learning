
#奇异谱分解--由colebrook于1978年首先在海洋学研究中提出并使用
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft
from sklearn.metrics import r2_score
from scipy.optimize import minimize


#####################################
#第1步：加载数据集、预处理
dataset = pd.read_csv("SAl_CEEMDR.csv", parse_dates=['date'], index_col=['date'])
dataset = dataset.dropna(axis=0, how ='any')
#data.drop(labels='Unnamed: 0', axis=1, inplace=True)
length = int(len(dataset))
print('length:',length)
series = dataset.A


# step1 嵌入--建立轨迹矩阵m*n
windowLen = 21  # 嵌入窗口长度L
seriesLen = len(series)  # 序列长度N
K = seriesLen - windowLen + 1
X = np.zeros((windowLen, K))
for i in range(K):
    X[:, i] = series[i:i + windowLen]

## step2: svd分解， U和sigma已经按升序排序
## 其中U是m*m左矩阵，sigma是奇异值,Vt是n*n右矩阵
U, sigma, VT = np.linalg.svd(X, full_matrices=False)

for i in range(VT.shape[0]):
    VT[i, :] *= sigma[i]
A = VT

## step3： 重组
rec = np.zeros((windowLen, seriesLen))
for i in range(windowLen):
    for j in range(windowLen - 1):
        for m in range(j + 1):
            rec[i, j] += A[i, j - m] * U[m, i]
        rec[i, j] /= (j+1)
    for j in range(windowLen - 1, seriesLen - windowLen + 1):
        for m in range(windowLen):
            rec[i, j] += A[i, j - m] * U[m, i]
        rec[i, j] /= windowLen
    for j in range(seriesLen - windowLen + 1, seriesLen):
        for m in range(j - seriesLen + windowLen, windowLen):
            rec[i, j] += A[i, j - m] * U[m, i]
        rec[i, j] /= (seriesLen - j)
## SSA处理的子序列和为rrr
rrr = np.sum(rec, axis=0)  #


#========画图法1=========
N = length  #
tMin, tMax = 0, N
T = np.linspace(tMin, tMax, N )
OK = 8

## 绘制原序列
plt.figure(figsize=(12, 18))
plt.subplots_adjust( bottom=None, top=None, wspace=0.1, hspace=0.7)
plt.subplot(OK+1, 1, 1)
plt.plot(T,series)
plt.ylabel("Raw-IMF1", fontsize='15', rotation=90, labelpad=15 )
plt.tick_params(labelsize=10)

## 绘制子序列
for i in range(OK):
    ax = plt.subplot(OK+1, 1, i + 2)
    ax.plot(T,rec[i, :], color='magenta')
    plt.ylabel("SSA-IMF" + str(i + 1), fontsize='15', rotation=90, labelpad=15 )
    plt.tick_params(labelsize=10)
plt.savefig('./AlCEE1-SSA.png')
plt.savefig('./AlCEE1-SSA.eps')
plt.show()



## 重构的检验
# r2检验，越接近1效果越好，负数表示完全没用...
score = r2_score(rrr, series[0:length])
print("r^2 值为： ", score)

# MAPE的实现
def N_MAPE(test_label1, y_predict):
    MAPE = np.mean(np.abs((y_predict - test_label1) / test_label1))*100
    return MAPE
print("MAPE:", N_MAPE(rrr, series[0:length] ) )


## step4: vstack
rrr1 = np.vstack((rec[0,:],rec[1,:],rec[2,:],rec[3,:],rec[4,:],rec[5,:],rec[6,:],rec[7,:]) )
print("rrr1:",rrr1)

# # 保存分组子序列文件
# df = pd.DataFrame(rrr1.T )
# df.to_csv("AlCEE1-SSA.csv", index=False, sep=',')


# # 保存子序列数据到文件中
# for i in range(8):
#     a = rec[i,: ]
#     dataframe = pd.DataFrame({'S{}'.format(i + 1): a} )
#     dataframe.to_csv("AlSSD-%d.csv" % (i + 1),  index=False, sep=',')


