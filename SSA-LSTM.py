
## *********
# 优化的模型
## *********
# 时间序列预测
import pandas as pd
import tensorflow as tf
import keras.backend.tensorflow_backend as K
import os
import numpy as np
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Dense, Input, Add, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.layers import concatenate  # 数组拼接
from tensorflow.keras.layers import GRU, LSTM, Bidirectional
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dropout
from scipy import signal
import openpyxl

import warnings
warnings.filterwarnings('ignore')

import time
start_time = time.clock()    # 程序开始时间
# function()   运行的程序

## ==****==
#第1步：加载数据集
dataset = pd.read_csv("SAl_CE1SSA.csv", parse_dates=['date'], index_col=['date'])
#data.drop(labels='Unnamed: 0', axis=1, inplace=True)


# 数据集描述
dataset.describe()

#第2步：数据集可视化
#第3步：数据预处理
# 删除多余的列 volume, code
#data.drop(columns=['volume', 'code'], axis=1, inplace=True)


# 分别对价格序列中{...}进行归一化
columns = ['A','B','C','D','E','F','G','H']
for col in columns:
    #最值归一化
    scaler = MinMaxScaler()
    dataset[col] = scaler.fit_transform(dataset[col].values.reshape(-1, 1)) #.


# 为了特征和标签的维度相同
# 将空值所在的行删除
dataset = dataset.dropna(axis=0, how ='any')


##========IMF1的处理===========##
# axis=1表示在行方向上操作（改变一行的长度）
# 特征数据集
x = dataset.drop(columns=['B','C','D','E','F','G','H'], axis=1)

# 标签数据集
y = dataset['A']

# 1 数据集分离： x_train, x_test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False, random_state=666)


## ============全局变量==========
Epoch = 100
batch_size = 16
# 定义最小周期-窗口大小
seq_len = 21

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
# if __name__ == '__main__':
#     m = 2
#     Data = data['A']
#     SE1 = SampEn(Data, m, r=0.2 * np.std(Data))
#     print("SE:", SE1)


##第4步：构造特征-标签
# 2 构造特征数据集
def create_dataset(x, y, seq_len):
    features = []
    targets = []

    for i in range(0, len(x) - seq_len, 1):
        data = x[i:i + seq_len]  # 序列数据
        label = y[i + seq_len]  # 标签数据
        # 保存到features和labels
        features.append(data)
        targets.append(label)
    # 返回
    return np.array(features), np.array(targets)

# ① 构造训练特征数据集
train_X, train_Y = create_dataset(x_train, y_train, seq_len)

# ② 构造测试特征数据集
test_X, test_Y = create_dataset(x_test, y_test, seq_len)

# 3 构造批数据
def create_batch_dataset(x, y, train=True, buffer_size=1000, batch_size=batch_size ):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y))) # 数据封装，tensor类型
    if train: # 训练集
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else: # 测试集
        return batch_data.batch(batch_size)

# 训练批数据
train_batch_dataset = create_batch_dataset(train_X, train_Y)

# 测试批数据
test_batch_dataset = create_batch_dataset(test_X, test_Y, train=False)


# 第五步-模型搭建
#with K.tf.device('/gpu:0'):
#np.random.seed(100)
visible1 = Input(shape=(seq_len, 1))  # 最后一数字表示特征数量个数
# 在权重参数w添加L2正则化
bi11 = LSTM(1, return_sequences=True)(visible1)
bi12 = LSTM(32,activation='selu', return_sequences=False)(bi11)
out1 = Dense(1)(bi12)
model = Model(inputs=[visible1], outputs=[out1])  # multi-input, multi-output


######################
# 打印训练集的特征形状
print('Shape of train_feature = ', train_X.shape)
# 显示模型结构
#plot_model(model, to_file="LSTM-ResNet.png", show_shapes=True)


## 回调函数 ReduceLROnPlateau 机制调整学习率
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=1,
                                            factor=0.1, #缩放学习率的值
                                            min_lr=1e-8 )
## 早停机制EarlyStopping
earlyStop = EarlyStopping(monitor='val_loss', #监测的值
                          min_delta=0, #增大或减小的阈值
                          patience=30, #容忍次数
                          mode='auto', #默认
                          verbose=1,  #日志显示函数
                          restore_best_weights=True)


# 定义优化器
opt = tf.keras.optimizers.Nadam(lr=1e-4 )

# 模型编译
# 其中，loss=mse， val_loss=val_mse
model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mape'])

## 模型在训练数据集上的拟合
History = model.fit(train_batch_dataset,
                    epochs=Epoch,
                    validation_data=test_batch_dataset,
                    callbacks=[learning_rate_reduction,earlyStop])

# 查看网络结构参数
model.summary()
# 查看history中参数
History.history.keys()


# 第6步：模型验证
test_preds = model.predict(test_X, verbose=1)
# 获取列值
test_preds = test_preds[:, 0].T #


##==============反归一化================
da = pd.read_csv("SAl_CE1SSA.csv", parse_dates=['date'], index_col=['date'])
#da.drop(columns=['time'], axis=1, inplace=True)
x_feature = da['A']  #对应的特征变量
# print(np.array(test_preds))
Var = np.zeros(test_preds.shape[0])
Var += (x_feature.max() - x_feature.min())*np.array(test_preds) + np.array(x_feature.min())


# # # 保存预测值到文件
# # ======= creating DataFrame by transforming Scalar Values to List=======
# dataframe = pd.DataFrame({'y_pred':Var })
# #将DataFrame存储为csv,index表示是否显示行名，default=True
# dataframe.to_csv("AlL-CE1SSAa.csv", index=False, sep=',')



# #========IMF2的处理===========##
## axis=1表示在行方向上操作（改变一行的长度）
# 特征数据集
xB = dataset.drop(columns=['A','C','D','E','F','G','H'], axis=1)

# 标签数据集
yB = dataset['B']

# 1 数据集分离： x_train, x_test
xB_train, xB_test, yB_train, yB_test = train_test_split(xB, yB, test_size=0.1, shuffle=False, random_state=666)

# ① 构造训练特征数据集
train_XB, train_YB = create_dataset(xB_train, yB_train, seq_len)

# ② 构造测试特征数据集
test_XB, test_YB = create_dataset(xB_test, yB_test, seq_len)

# 训练批数据
train_batch_datasetB = create_batch_dataset(train_XB, train_YB)

# 测试批数据
test_batch_datasetB = create_batch_dataset(test_XB, test_YB, train=False)


# 第五步-模型搭建
#with K.tf.device('/gpu:0'):
#np.random.seed(100)
modelB = Sequential([
    LSTM(units=1, input_shape=(seq_len, 1),return_sequences=True), #1
    LSTM(units=32, input_shape=(seq_len, 1),activation='selu', return_sequences=False), #2
    Dense(1)
])


#########################################
# 打印训练集的特征形状
print('Shape of train_feature = ', train_XB.shape)
# 显示模型结构
#plot_model(model, to_file="LSTM-ResNet.png", show_shapes=True)

# 定义优化器
optimizer = tf.keras.optimizers.Nadam(lr=1e-2 )

# 模型编译
modelB.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])

# 模型训练
HistoryB = modelB.fit(train_batch_datasetB,
                  epochs=Epoch,
                  validation_data=test_batch_datasetB,
                  callbacks=[learning_rate_reduction,earlyStop])
# 查看history中参数
HistoryB.history.keys()


# 第6步：模型验证
test_predsB = modelB.predict(test_XB, verbose=1)
# 获取列值
test_predsB = test_predsB[:, 0].T #


##==============反归一化================
#da.drop(columns=['time'], axis=1, inplace=True)
x_featureB = da['B']  #对应的特征变量
# print(np.array(test_preds))
VarB = np.zeros(test_preds.shape[0])
VarB += (x_featureB.max() - x_featureB.min())*np.array(test_predsB) + np.array(x_featureB.min())


# # ======= creating DataFrame by transforming Scalar Values to List=======
# dataframe = pd.DataFrame({'y_pred':VarB })
# dataframe.to_csv("AlL-CE1SSAb.csv", index=False, sep=',')



##========IMF3的处理===========##
# 特征数据集
xC = dataset.drop(columns=['A','B','D','E','F','G','H'], axis=1)

# 标签数据集
yC = dataset['C']

# 1 数据集分离： x_train, x_test
xC_train, xC_test, yC_train, yC_test = train_test_split(xC, yC, test_size=0.1, shuffle=False, random_state=666)


# ① 构造训练特征数据集
train_XC, train_YC = create_dataset(xC_train, yC_train, seq_len)

# ② 构造测试特征数据集
test_XC, test_YC = create_dataset(xC_test, yC_test, seq_len)

# 训练批数据
train_batch_datasetC = create_batch_dataset(train_XC, train_YC)

# 测试批数据
test_batch_datasetC = create_batch_dataset(test_XC, test_YC, train=False)


# 第五步-模型搭建
#with K.tf.device('/gpu:0'):
#np.random.seed(100)
modelC = Sequential([
    LSTM(units=1, input_shape=(seq_len, 1),return_sequences=True), #1
    LSTM(units=32, input_shape=(seq_len, 1),activation='selu', return_sequences=False), #2
    Dense(1)
])


######################
# 打印训练集的特征形状
print('Shape of train_feature = ', train_XC.shape)
# 显示模型结构
#plot_model(model, to_file="LSTM-ResNet.png", show_shapes=True)


# 模型编译
modelC.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])

# 模型训练
HistoryC = modelC.fit(train_batch_datasetC,
                  epochs=Epoch,
                  validation_data=test_batch_datasetC,
                  callbacks=[learning_rate_reduction,earlyStop])
# 查看history中参数
HistoryC.history.keys()


# 第6步：模型验证
test_predsC = modelC.predict(test_XC, verbose=1)
# 获取列值
test_predsC = test_predsC[:, 0].T #


##==============反归一化================
#da.drop(columns=['time'], axis=1, inplace=True)
x_featureC = da['C']  #对应的特征变量
# print(np.array(test_preds))
VarC = np.zeros(test_preds.shape[0])
VarC += (x_featureC.max() - x_featureC.min())*np.array(test_predsC) + np.array(x_featureC.min())


# # ======= creating DataFrame by transforming Scalar Values to List=======
# dataframe = pd.DataFrame({'y_pred':VarC })
# dataframe.to_csv("AlL-CE1SSAc.csv", index=False, sep=',')



##========IMF4的处理===========##
# 特征数据集
xD = dataset.drop(columns=['A','B','C','E','F','G','H'], axis=1)

# 标签数据集
yD = dataset['D']

# 1 数据集分离： x_train, x_test
xD_train, xD_test, yD_train, yD_test = train_test_split(xD, yD, test_size=0.1, shuffle=False, random_state=666)

# ① 构造训练特征数据集
train_XD, train_YD = create_dataset(xD_train, yD_train, seq_len)

# ② 构造测试特征数据集
test_XD, test_YD = create_dataset(xD_test, yD_test, seq_len)


# 训练批数据
train_batch_datasetD = create_batch_dataset(train_XD, train_YD)

# 测试批数据
test_batch_datasetD = create_batch_dataset(test_XD, test_YD, train=False)


# 第五步-模型搭建
#with K.tf.device('/gpu:0'):
#np.random.seed(100)
modelD = Sequential([
    LSTM(units=1, input_shape=(seq_len, 1),return_sequences=True), #1
    LSTM(units=32, input_shape=(seq_len, 1),activation='selu', return_sequences=False), #2
    Dense(1)
])


#############################
# 打印训练集的特征形状
print('Shape of train_feature = ', train_XD.shape)
# 显示模型结构
#plot_model(model, to_file="LSTM-ResNet.png", show_shapes=True)


# 模型编译
modelD.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])

# 模型训练
HistoryD = modelD.fit(train_batch_datasetD,
                  epochs=Epoch,
                  validation_data=test_batch_datasetD,
                  callbacks=[learning_rate_reduction,earlyStop])
# 查看history中参数
HistoryD.history.keys()


# 第6步：模型验证
test_predsD = modelD.predict(test_XD, verbose=1)
# 获取列值
test_predsD = test_predsD[:, 0].T


##==============反归一化================
#da.drop(columns=['time'], axis=1, inplace=True)
x_featureD = da['D']  #对应的特征变量
# print(np.array(test_preds))
VarD = np.zeros(test_predsD.shape[0])
VarD += (x_featureD.max() - x_featureD.min())*np.array(test_predsD) + np.array(x_featureD.min())

# # ======= creating DataFrame by transforming Scalar Values to List=======
# dataframe = pd.DataFrame({'y_pred':VarD })
# dataframe.to_csv("AlL-CE1SSAd.csv", index=False, sep=',')



##========IMF5的处理===========##
# 特征数据集
xE = dataset.drop(columns=['A','B','C','D','F','G','H'], axis=1)

# 标签数据集
yE = dataset['E']

# 1 数据集分离： x_train, x_test
xE_train, xE_test, yE_train, yE_test = train_test_split(xE, yE, test_size=0.1, shuffle=False, random_state=666)


# ① 构造训练特征数据集
train_XE, train_YE = create_dataset(xE_train, yE_train, seq_len)

# ② 构造测试特征数据集
test_XE, test_YE = create_dataset(xE_test, yE_test, seq_len)

# 训练批数据
train_batch_datasetE = create_batch_dataset(train_XE, train_YE)

# 测试批数据
test_batch_datasetE = create_batch_dataset(test_XE, test_YE, train=False)


# 第五步-模型搭建
#with K.tf.device('/gpu:0'):
#np.random.seed(100)
modelE = Sequential([
    LSTM(units=1, input_shape=(seq_len, 1),return_sequences=True), #1
    LSTM(units=32, input_shape=(seq_len, 1),activation='selu', return_sequences=False), #2
    Dense(1)
])


#######################################
# 打印训练集的特征形状
print('Shape of train_feature = ', train_XE.shape)
# 显示模型结构
#plot_model(model, to_file="LSTM-ResNet.png", show_shapes=True)


# 模型编译
# 其中，loss=mse， val_loss=val_mse
modelE.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])


# 模型训练
HistoryE = modelE.fit(train_batch_datasetE,
                  epochs=Epoch,
                  validation_data=test_batch_datasetE,
                  callbacks=[learning_rate_reduction,earlyStop])
# 查看history中参数
HistoryE.history.keys()


# 第6步：模型验证
test_predsE = modelE.predict(test_XE, verbose=1)
# 获取列值
test_predsE = test_predsE[:, 0].T #


##==============反归一化================
#da.drop(columns=['time'], axis=1, inplace=True)
x_featureE = da['E']  #对应的特征变量
# print(np.array(test_preds))
VarE = np.zeros(test_predsE.shape[0])
VarE += (x_featureE.max() - x_featureE.min())*np.array(test_predsE) + np.array(x_featureE.min())


# # ======= creating DataFrame by transforming Scalar Values to List=======
# dataframe = pd.DataFrame({'y_pred':VarE })
# dataframe.to_csv("AlL-CE1SSAe.csv", index=False, sep=',')



##========IMF6的处理===========##
# 特征数据集
xF = dataset.drop(columns=['A','B','C','D','E','G','H'], axis=1)

# 标签数据集
yF = dataset['F']

# 1 数据集分离： x_train, x_test
xF_train, xF_test, yF_train, yF_test = train_test_split(xF, yF, test_size=0.1, shuffle=False, random_state=666)


# ① 构造训练特征数据集
train_XF, train_YF = create_dataset(xF_train, yF_train, seq_len)

# ② 构造测试特征数据集
test_XF, test_YF = create_dataset(xF_test, yF_test, seq_len)


# 训练批数据
train_batch_datasetF = create_batch_dataset(train_XF, train_YF)

# 测试批数据
test_batch_datasetF = create_batch_dataset(test_XF, test_YF, train=False)


# 第五步-模型搭建
#with K.tf.device('/gpu:0'):
#np.random.seed(100)
modelF = Sequential([
    LSTM(units=1, input_shape=(seq_len, 1),return_sequences=True), #1
    LSTM(units=32, input_shape=(seq_len, 1),activation='selu', return_sequences=False), #2
    Dense(1)
])


# 模型编译
modelF.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])

# 模型训练
HistoryF = modelF.fit(train_batch_datasetF,
                  epochs=Epoch,
                  validation_data=test_batch_datasetF,
                  callbacks=[learning_rate_reduction,earlyStop])
# 查看history中参数
HistoryF.history.keys()


# 第6步：模型验证
test_predsF = modelF.predict(test_XF, verbose=1)
# 获取列值
test_predsF = test_predsF[:, 0].T #


##==============反归一化================
#da.drop(columns=['time'], axis=1, inplace=True)
x_featureF = da['F']  #对应的特征变量
# print(np.array(test_preds))
VarF = np.zeros(test_predsF.shape[0])
VarF += (x_featureF.max() - x_featureF.min())*np.array(test_predsF) + np.array(x_featureF.min())

# # ======= creating DataFrame by transforming Scalar Values to List=======
# dataframe = pd.DataFrame({'y_pred':VarF })
# dataframe.to_csv("AlL-CE1SSAf.csv", index=False, sep=',')
#

##=======IMF7==========##
# 特征数据集
xG = dataset.drop(columns=['A','B','C','D','E','F','H'], axis=1)

# 标签数据集
yG = dataset['G']

# 1 数据集分离： x_train, x_test
xG_train, xG_test, yG_train, yG_test = train_test_split(xG, yG, test_size=0.1, shuffle=False, random_state=666)


# ① 构造训练特征数据集
train_XG, train_YG = create_dataset(xG_train, yG_train, seq_len)

# ② 构造测试特征数据集
test_XG, test_YG = create_dataset(xG_test, yG_test, seq_len)

# 3 构造批数据
# 训练批数据
train_batch_datasetG = create_batch_dataset(train_XG, train_YG)

# 测试批数据
test_batch_datasetG = create_batch_dataset(test_XG, test_YG, train=False)


# # 第五步-模型搭建
#with K.tf.device('/gpu:0'):
#np.random.seed(100)
modelG = Sequential([
    LSTM(units=1, input_shape=(seq_len, 1),return_sequences=True), #1
    LSTM(units=32, input_shape=(seq_len, 1),activation='selu', return_sequences=False), #2
    Dense(1)
])


##############################
# 打印训练集的特征形状
print('Shape of train_feature = ', train_XG.shape)
# 显示模型结构
#plot_model(model, to_file="LSTM-ResNet.png", show_shapes=True)


## 定义优化器
# 模型编译
modelG.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])

# 模型训练
HistoryG = modelG.fit(train_batch_datasetG,
                  epochs=Epoch,
                  validation_data=test_batch_datasetG,
                  callbacks=[learning_rate_reduction,earlyStop])
# 查看history中参数
HistoryG.history.keys()


# 第6步：模型验证
test_predsG = modelG.predict(test_XG, verbose=1)
# 获取列值
test_predsG = test_predsG[:, 0].T #


##==============反归一化================
#da.drop(columns=['time'], axis=1, inplace=True)
x_featureG = da['G']  #对应的特征变量
# print(np.array(test_preds))
VarG = np.zeros(test_predsG.shape[0])
VarG += (x_featureG.max() - x_featureG.min())*np.array(test_predsG) + np.array(x_featureG.min())


# # ======= creating DataFrame by transforming Scalar Values to List=======
# dataframe = pd.DataFrame({'y_pred':VarG })
# dataframe.to_csv("AlL-CE1SSAg.csv", index=False, sep=',')


##=======IMF8预测========##
# 特征数据集
xH = dataset.drop(columns=['A','B','C','D','E','F','G'], axis=1)

# 标签数据集
yH = dataset['H']

# 1 数据集分离： x_train, x_test
xH_train, xH_test, yH_train, yH_test = train_test_split(xH, yH, test_size=0.1, shuffle=False, random_state=666)


# ① 构造训练特征数据集
train_XH, train_YH = create_dataset(xH_train, yH_train, seq_len)

# ② 构造测试特征数据集
test_XH, test_YH = create_dataset(xH_test, yH_test, seq_len)

# 3 构造批数据
# 训练批数据
train_batch_datasetH = create_batch_dataset(train_XH, train_YH)

# 测试批数据
test_batch_datasetH = create_batch_dataset(test_XH, test_YH, train=False)


# 第五步-模型搭建
#with K.tf.device('/gpu:0'):
#np.random.seed(100)
modelH = Sequential([
    LSTM(units=1, input_shape=(seq_len, 1),return_sequences=True), #1
    LSTM(units=32, input_shape=(seq_len, 1),activation='selu', return_sequences=False), #2
    Dense(1)
])


################################
# 打印训练集的特征形状
print('Shape of train_feature = ', train_XH.shape)
# 显示模型结构
#plot_model(model, to_file="LSTM-ResNet.png", show_shapes=True)


# 模型编译
# 其中，loss=mse， val_loss=val_mse
modelH.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])


# 模型训练
HistoryH = modelH.fit(train_batch_datasetH,
                  epochs=Epoch,
                  validation_data=test_batch_datasetH,
                  callbacks=[learning_rate_reduction,earlyStop])

# 查看history中参数
HistoryH.history.keys()


# 第6步：模型验证
test_predsH = modelH.predict(test_XH, verbose=1)
# 获取列值
test_predsH = test_predsH[:, 0].T #
end_time = time.clock()    # end
run_time = end_time - start_time    #
print('run_time:',run_time)


##==============反归一化================
#da.drop(columns=['time'], axis=1, inplace=True)
x_featureH = da['H']  #对应的特征变量
# print(np.array(test_preds))
VarH= np.zeros(test_predsH.shape[0])
VarH += (x_featureH.max() - x_featureH.min())*np.array(test_predsH) + np.array(x_featureH.min())


# # ======= creating DataFrame by transforming Scalar Values to List=======
# dataframe = pd.DataFrame({'y_pred':VarH })
# dataframe.to_csv("AlL-CE1SSAh.csv", index=False, sep=',')

## ++++合并预测++++
variance = np.array(Var) +np.array(VarB) +np.array(VarC) +np.array(VarD) +np.array(VarE) +np.array(VarF) \
        +np.array(VarG) +np.array(VarH)
# # 保存预测值到文件
pred_data = pd.DataFrame({'y_pred':variance })
#将DataFrame存储为csv,index表示是否显示行名，default=True
pred_data.to_csv("AlL_CE1SSA.csv",  index=False, sep=',')

## ++++vstack format++++
r = np.vstack((Var,VarB,VarC,VarD,VarE,VarF,VarG,VarH) )
# 保存分组子序列文件
df = pd.DataFrame(r.T)
df.to_csv("AlL_CE1SSA0.csv", index=False, sep=',')

## ++++metrics--r2_score++++
print("metrics of ABCD:\n", r2_score(test_Y, test_preds),r2_score(test_YB, test_predsB),r2_score(test_YC, test_predsC),r2_score(test_YD, test_predsD) )
print("metrics of EFGH:\n", r2_score(test_YE, test_predsE),r2_score(test_YF, test_predsF),r2_score(test_YG, test_predsG),r2_score(test_YH, test_predsH) )








