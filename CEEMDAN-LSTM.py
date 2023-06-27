
#导入必要的库
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain  #二维列表转为一维
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense, Input, Add, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.layers import concatenate  # 数组拼接
from tensorflow.keras.layers import GRU, LSTM, Bidirectional
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
import keras.backend.tensorflow_backend as K


import time
start_time = time.clock()    # 程序开始时间
# function()   运行的程序

#####################
#第1步：加载数据集、预处理
dataset = pd.read_csv("SAl_CEEMDR.csv", parse_dates=['date'], index_col=['date'])
#dataset.drop(labels='Unnamed: 0', axis=1, inplace=True)


# 数据集描述
dataset.describe()

#第2步：数据集可视化
#第3步：数据预处理
# 删除多余的列 volume, code
#dataset.drop(columns=['volatility'], axis=1, inplace=True)


# 分别对价格序列中{...}进行归一化
columns = ['A','B','C','IMFs']
for col in columns:
    #最值归一化
    scaler = MinMaxScaler()
    dataset[col] = scaler.fit_transform(dataset[col].values.reshape(-1, 1)) #


# 为了特征和标签的维度相同
# 将空值所在的行删除
dataset = dataset.dropna(axis=0, how ='any')


# axis=1表示在行方向上操作（改变一行的长度）
# 特征数据集
x = dataset.drop(columns=['B','C','IMFs'], axis=1)

# 标签数据集
y = dataset['A']

# 1 数据集分离： x_train, x_test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False, random_state=666)


##===============全局变量=========================
Epoch = 100
batch_size = 16
# 定义最小周期-窗口大小
seq_len = 21

##第4步：构造特征-标签
# 2 构造特征数据集
def create_dataset(x, y, seq_len):
    features = []
    targets = []

    for i in range(0, len(x) - seq_len, 1):
        data = x[i:i + seq_len]  # 特征数据
        label = y[i + seq_len]  # 标签数据
        # 保存到features和labels
        features.append(data)
        targets.append(label)
    # 返回
    return np.array(features), np.array(targets)

# ① 构造训练特征数据集
train_feature, train_labels = create_dataset(x_train, y_train, seq_len)

# ② 构造测试特征数据集
test_feature, test_labels = create_dataset(x_test, y_test, seq_len)

# 3 构造批数据
def create_batch_dataset(x, y, train=True, buffer_size=1000, batch_size=batch_size ):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y))) # 数据封装，tensor类型
    if train: # 训练集
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else: # 测试集
        return batch_data.batch(batch_size)

# 训练批数据
train_batch_dataset = create_batch_dataset(train_feature, train_labels)

# 测试批数据
test_batch_dataset = create_batch_dataset(test_feature, test_labels, train=False)

##第5步：BP模型搭建、编译、训练
# 模型搭建--版本2
model = Sequential([
    LSTM(units=1, input_shape=(seq_len, 1),return_sequences=True), #1
    LSTM(units=32, input_shape=(seq_len, 1),activation='selu', return_sequences=False), #2
    Dense(1)
])


#################
# 打印训练集的特征形状
print('Shape of train_feature = ', train_feature.shape)
# 显示模型结构
#plot_model(model, to_file="BP.png", show_shapes=True)

#modelfileL = 'modelweight.model' #权重保存文件
## 回调函数ReduceLROnPlateau 机制调整学习率
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=20, verbose=1,
                                            factor=0.1,  # 缩放学习率的值
                                            min_lr=1e-10)
## 早停机制EarlyStopping
earlyStop = EarlyStopping(monitor='loss',  # 监测的值
                          min_delta=0,  # 增大或减小的阈值
                          patience=30,  # 容忍次数
                          mode='auto',  # 默认
                          verbose=1,  # 日志显示函数
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


#model.save_weights(modelfileL) #保存模型权重
# 查看网络结构参数
model.summary()
# 查看history中存储了哪些参数
History.history.keys()


# 第6步：模型验证
test_preds = model.predict(test_feature, verbose=1)
# 获取列值
test_preds = test_preds[:, 0].T
# test_preds = list(chain.from_iterable(test_preds))
# test_preds = np.array(test_preds)



##==============反归一化================
da = pd.read_csv("SAl_CEEMDR.csv", parse_dates=['date'], index_col=['date'])
#da.drop(columns=['time'], axis=1, inplace=True)
x_feature = da['A']  #对应的特征变量
#print('np.array(test_preds)',np.array(test_preds))
Var = np.zeros(test_preds.shape )
Var += (x_feature.max() -x_feature.min()) *test_preds +np.array(x_feature.min())


# 保存预测值到文件
# creating DataFrame by transforming Scalar Values to List
# dataframe = pd.DataFrame({'y_pred':Var })
# #将DataFrame存储为csv,index表示是否显示行名，default=True
# dataframe.to_csv("Ala.csv", index=False, sep=',')



# #========IMF2的处理===========##
# 特征数据集
xB = dataset.drop(columns=['A','C','IMFs'], axis=1)

# 标签数据集
yB = dataset['B']

# 1 数据集分离： x_train, x_test
xB_train, xB_test, yB_train, yB_test = train_test_split(xB, yB, test_size=0.1, shuffle=False, random_state=666)


# ① 构造训练特征数据集
train_XB, train_YB = create_dataset(xB_train, yB_train, seq_len)

# ② 构造测试特征数据集
test_XB, test_YB = create_dataset(xB_test, yB_test, seq_len)

# 3 构造批数据

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

# 查看网络结构参数
modelB.summary()
# 查看history中存储了哪些参数
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



##========IMF3的处理===========##
# 特征数据集
xC = dataset.drop(columns=['A','B','IMFs'], axis=1)

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


# 显示模型结构
#plot_model(model, to_file="LSTM-ResNet.png", show_shapes=True)

# # 定义优化器
# optC = tf.keras.optimizers.Nadam(lr=1e-2 )

# 模型编译
modelC.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])

# 模型训练
HistoryC = modelC.fit(train_batch_datasetC,
                  epochs=Epoch,
                  validation_data=test_batch_datasetC,
                  callbacks=[learning_rate_reduction,earlyStop])

# 查看网络结构参数
modelC.summary()
# 查看history中存储了哪些参数
HistoryC.history.keys()


# 第6步：模型验证
test_predsC = modelC.predict(test_XC, verbose=1)
# 获取列值
test_predsC = test_predsC[:, 0].T


##==============反归一化================
#da.drop(columns=['time'], axis=1, inplace=True)
x_featureC = da['C']  #
# print(np.array(test_preds))
VarC = np.zeros(test_preds.shape[0])
VarC += (x_featureC.max() - x_featureC.min())*np.array(test_predsC) + np.array(x_featureC.min())



##========IMF4的处理===========##
# 特征数据集
xD = dataset.drop(columns=['A','B','C'], axis=1)

# 标签数据集
yD = dataset['IMFs']

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


# 显示模型结构
#plot_model(model, to_file="LSTM-ResNet.png", show_shapes=True)

# # 定义优化器
# optD = tf.keras.optimizers.Nadam(lr=1e-2 )

# 模型编译
modelD.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])

# 模型训练
HistoryD = modelD.fit(train_batch_datasetD,
                  epochs=Epoch,
                  validation_data=test_batch_datasetD,
                  callbacks=[learning_rate_reduction,earlyStop])

# 查看网络结构参数
modelD.summary()
# 查看history中存储了哪些参数
HistoryD.history.keys()


# 第6步：模型验证
test_predsD = modelD.predict(test_XD, verbose=1)
# 获取列值
test_predsD = test_predsD[:, 0].T
end_time = time.clock()    # end running
run_time = end_time - start_time    #
print('run_time:',run_time)



##==============反归一化================
#da.drop(columns=['time'], axis=1, inplace=True)
x_featureD = da['IMFs']  #对应的特征变量
# print(np.array(test_preds))
VarD = np.zeros(test_predsD.shape[0])
VarD += (x_featureD.max() - x_featureD.min())*np.array(test_predsD) + np.array(x_featureD.min())



## ++++合并预测++++
variance = np.array(Var) +np.array(VarB) +np.array(VarC)  +np.array(VarD) # Predicted value
# # 保存预测值到文件
pred_data = pd.DataFrame({'y_pred':variance  })
#将DataFrame存储为csv,index表示是否显示行名，default=True
pred_data.to_csv("AlCEE-L.csv",  index=False, sep=',')

## ++++局部合并预测++++
final1 = np.array(VarB) +np.array(VarC) +np.array(VarD)
# # 保存预测值到文件
pred_d = pd.DataFrame({'y_pred':final1  })
pred_d.to_csv("AlCEE-L-BCD.csv",  index=False, sep=',')



#=======与原始数据比较=======
df = pd.read_csv("SHFE_Al.csv", parse_dates=['date'], index_col=['date'])
#df.drop(labels='Unnamed: 0', axis=1, inplace=True)

# 特征数据集
x1 = df.drop(columns=['N_price'], axis=1)

# 标签数据集
y1 = df['close']


# 1 数据集分离： x_train, x_test
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.1, shuffle=False, random_state=666)

# ① 构造训练特征数据集
Train_feature, Train_Label = create_dataset(x1_train, y1_train, seq_len)

# ② 构造测试特征数据集
Test_feature, Test_Labels = create_dataset(x1_test, y1_test, seq_len)


## Define metrics
# RMSE的定义
def N_RMSE(test_label1, y_predict):
    RMSE = metrics.mean_squared_error(test_label1, y_predict)**0.5
    return RMSE


# MSE的定义
def N_MSE(test_label1, y_predict):
    MSE = metrics.mean_squared_error(test_label1, y_predict)
    return MSE

#MAE的定义
def N_MAE(test_label1, y_predict):
    MAE = metrics.mean_absolute_error(test_label1, y_predict)
    return MAE

# MAPE的实现
def N_MAPE(test_label1, y_predict):
    MAPE = np.mean(np.abs((y_predict - test_label1) / test_label1))*100
    return MAPE

lth = int(len(dataset))
long = int(len(Test_Labels))
print("metrics of IMF1:\n", N_RMSE(x_feature[lth-long:lth], Var ), N_MAE(x_feature[lth-long:lth], Var ), N_MAPE(x_feature[lth-long:lth], Var) )

print("metrics of IMF2:\n", N_RMSE(x_featureB[lth-long:lth], VarB ), N_MAE(x_featureB[lth-long:lth], VarB), N_MAPE(x_featureB[lth-long:lth], VarB) )

print("metrics of IMF3:\n", N_RMSE(x_featureC[lth-long:lth], VarC ), N_MAE(x_featureC[lth-long:lth], VarC), N_MAPE(x_featureC[lth-long:lth], VarC) )

print("metrics of IMF4:\n", N_RMSE(x_featureD[lth-long:lth], VarD ), N_MAE(x_featureD[lth-long:lth], VarD), N_MAPE(x_featureD[lth-long:lth], VarD) )


print("The metrics of predicted value:" )
print('r2_score', r2_score(Test_Labels, variance))
print("RMSE:", N_RMSE(Test_Labels, variance ) )
#print("MSE:", N_MSE(Test_labels, variance ) )
print("MAE:", N_MAE(Test_Labels, variance ) )
print("MAPE:", N_MAPE(Test_Labels, variance ) )


# ===========+ Visualization +=============
plt.figure(figsize=(12,6))
plt.plot(Test_Labels, label="Original data")
plt.plot(variance, label="Forecast data")
plt.legend( loc='best', fontsize='20')
# setting labelsize（x axis and y axis）
plt.tick_params(labelsize=15)

## setting distance to axis
plt.tick_params(pad = 0.03)  #
plt.xlabel("Trade day", fontsize=20)
plt.ylabel("Forecast price (CNY/kg)", fontsize=20) #
plt.title("Test Data", fontsize=20)
#plt.savefig('./Zn-LR.jpg')
plt.show()




