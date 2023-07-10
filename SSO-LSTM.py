import os
import SSO  #已定义的麻雀搜索算法
import matplotlib
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Add, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.layers import GRU, LSTM, Bidirectional

from tensorflow.keras.optimizers import Nadam,SGD, Adam

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import warnings
warnings.filterwarnings('ignore')


###################
#第1步：加载数据集
dataset = pd.read_csv("SHFE_Al.csv", parse_dates=['date'], index_col=['date'])
#dataset.drop(labels='Unnamed: 0', axis=1, inplace=True)


# 数据集信息
dataset.info()

# 数据集描述
dataset.describe()

#第2步：数据集可视化
#第3步：数据预处理
# 删除多余的列 volume, code
#dataset.drop(columns=['volatility'], axis=1, inplace=True)

# 分别对价格序列中{close，open, high, low}进行归一化
columns = ['close','N_price']
for col in columns:
    #最值归一化
    scaler = MinMaxScaler()
    dataset[col] = scaler.fit_transform(dataset[col].values.reshape(-1, 1)) #


# 为了特征和标签的维度相同
# 将空值所在的行删除
dataset = dataset.dropna(axis=0, how ='any')

# axis=1表示在行方向上操作（改变一行的长度）
# 特征数据集
x = dataset.drop(columns=['N_price'], axis=1)

# 标签数据集
y = dataset['close']

# 1 数据集分离： x_train, x_test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False, random_state=666)


##=====全局变量========
Epoch = 10
batch_size = 16  #
# 定义最小周期-窗口大小
seq_len = 21


##第4步：构造特征-标签
# 2 构造特征数据集
def create_dataset(X, y, seq_len):
    features = []
    targets = []  # 标签

    for i in range(0, len(X) - seq_len, 1):  # 此处的1表示步长，每隔一步滑一下
        data = X.iloc[i:i + seq_len]  # 序列数据；前闭后开
        label = y.iloc[i + seq_len]  # 标签数据
        # 保存到features和labels
        features.append(data)
        targets.append(label)
    # 返回
    return np.array(features), np.array(targets)

# ① 构造训练特征数据集
train_feature, train_label = create_dataset(x_train, y_train, seq_len)

# ② 构造测试特征数据集
test_feature, test_label = create_dataset(x_test, y_test, seq_len)

##3 构造批数据
def create_batch_dataset(x, y, train=True, buffer_size=1000, batch_size=batch_size ):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y))) # 数据封装，tensor类型
    if train: # 训练集
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else: # 测试集
        return batch_data.batch(batch_size)

# 训练批数据
train_batch_dataset = create_batch_dataset(train_feature, train_label)

# 测试批数据
test_batch_dataset = create_batch_dataset(test_feature, test_label, train=False)


# 第五步-模型搭建
def build_model(neurons1,learn_rate):
    model = Sequential([
        LSTM(units=neurons1, input_shape=(seq_len, 1), activation='selu', return_sequences=False),  # 1
        # LSTM(units=16,  return_sequences=True), #2
        # layers.Dropout(0.3),
        # LSTM(units=16, kernel_regularizer=regularizers.l2(0.01), activation='selu'),
        Dense(1)
    ])

    # 设置优化器
    optimizer = Nadam(lr=learn_rate)

    # 模型编译
    # 其中，loss=mse， val_loss=val_mse
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])  # mse, mae, mape
    return model


def training(X):
    neurons1 = int(X[0])
    #neurons2 = int(X[1])
    #neurons3 = int(X[2])
    #dropout = round(X[2], 6)
    learn_rate = X[1]
    #decay_rate = X[2]
    #c = X[2]
    #batch_size = int(X[5])
    print(X)
    model = build_model(neurons1,learn_rate )

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


    ## 模型在训练数据集上的拟合
    model.fit(train_feature, train_label,
              epochs=Epoch,
              verbose=1,
              # batch_size=batch_size,
              callbacks=[learning_rate_reduction, earlyStop])

    ##第6步：模型验证
    pred = model.predict(test_feature)
    temp_mse = mean_squared_error(test_label, pred)
    print('MSE:',temp_mse)
    return temp_mse


if __name__ == '__main__':
    '''
    神经网络第一层神经元个数
    神经网络第二层神经元个数
    dropout比率
    batch_size
    '''
    up = [128, 0.01]  #上边界
    down = [16, 1e-5]  #下边界
    # 设置参数
    pop_size = 5  # 种群数量
    n_dim = 2  # 维度
    max_iter = 12  # 最大迭代次数
    # 适应度函数选择
    fobj = training

    # 开始优化
    GbestScore,GbestPositon,Curve = SSO.SSA(fobj, pop_size, n_dim, max_iter, lb=down, ub=up)
    print('最优适应度值：', GbestScore)
    print('最优解：', GbestPositon)

    ## ++++++绘制适应度曲线+++++++
    x0 = range(1, 13, 1)
    plt.figure(figsize=(12,6) )
    # 调整子图布局
    plt.subplots_adjust(bottom=None, top=None, wspace=0.5, hspace=0.5)
    ## 使用semilogy()函数，将变化放大
    plt.semilogy(x0, Curve, color='slateblue', linestyle='--', marker='o', markerfacecolor='w')
    plt.xlabel('Iteration', fontsize='20')
    plt.ylabel("Fitness", fontsize='20')
    # 刻度值字体大小设置（x轴和y轴同时设置）
    plt.tick_params(labelsize=15)
    plt.grid()
    plt.title('SSO convergence curve', fontsize='20')
    plt.savefig('./c-SSO-LSTM.jpg')
    plt.show()


    ## 训练模型---使用PSO找到的最好的神经元个数
    # neurons1 = int(GbestPositon[0])
    # neurons2 = int(GbestPositon[1])
    # neurons3 = int(GbestPositon[2])
    # learn_rate = GbestPositon[3]
    # decay_rate = GbestPositon[4]
    # batch_size = int(GbestPositon[5])
    # model = build_model(neurons1, neurons2, neurons3, learn_rate, decay_rate)
    #
    # # 保存模型权重文件和训练日志
    # log_file = os.path.join('logs00')
    # tensorboard_callback = TensorBoard(log_file)
    # checkpoint_file = "best_model.hdf500"
    # checkpoint_callback = ModelCheckpoint(filepath=checkpoint_file,
    #                                       monitor='loss',
    #                                       mode='min',
    #                                       save_best_only=True,
    #                                       save_weights_only=False)
    #
    # # 模型训练
    # History = model.fit(train_feature, train_label,
    #                     epochs=Epoch,
    #                     validation_split=0.1,
    #                     verbose=1,
    #                     batch_size=batch_size,
    #                     callbacks=[tensorboard_callback, checkpoint_callback])




