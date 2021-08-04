import os
import datetime
import time

import wfdb
import pywt
import seaborn
import numpy as np
from pylab import *
from matplotlib.font_manager import FontProperties
# import tensorflow as tf
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 项目目录
project_path = "/home/root/mit-bih_ecg_recognition/"
# 定义日志目录,必须是启动web应用时指定目录的子目录,建议使用日期时间作为子目录名
log_dir = project_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
test_model_path = "ecg_model.tflite"

# 测试集在数据集中所占的比例
RATIO = 0.3


# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data, Y_data):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']

    # 读取心电数据记录
    print("Reading No. " + number + " heatbeat data...")
    record = wfdb.rdrecord('ecg_data/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('ecg_data/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    while i < j:
        try:
            lable = ecgClassSet.index(Rclass[i])
            x_train = rdata[Rlocation[i] - 99:Rlocation[i] + 201]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


# 加载数据集并进行预处理
def loadData():
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    lableSet = []
    for n in numberSet:
        getDataSet(n, dataSet, lableSet)

    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, 300)
    lableSet = np.array(lableSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, lableSet))
    np.random.shuffle(train_ds)

    # 数据集及其标签集
    X = train_ds[:, :300].reshape(-1, 300, 1)
    Y = train_ds[:, 300]

    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    test_length = int(RATIO * len(shuffle_index))
    test_index = shuffle_index[:test_length]
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    return X_train, Y_train, X_test, Y_test


# 构建CNN模型
""" def buildModel():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(300, 1)),
        # 第一个卷积层, 4 个 21x1 卷积核
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='SAME', activation='relu'),
        # 第一个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第二个卷积层, 16 个 23x1 卷积核
        tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='SAME', activation='relu'),
        # 第二个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第三个卷积层, 32 个 25x1 卷积核
        tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='SAME', activation='relu'),
        # 第三个池化层, 平均池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第四个卷积层, 64 个 27x1 卷积核
        tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='SAME', activation='relu'),
        # 打平层,方便全连接层处理
        tf.keras.layers.Flatten(),
        # 全连接层,128 个节点
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout层,dropout = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        # 全连接层,5 个节点
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return newModel """


# 混淆矩阵
# Y_data将NAVLR按顺序转换为01234
def plotHeatMap(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    # 归一化
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    # con_mat_norm = np.around(con_mat_norm, decimals=2)

    fname = '/home/root/MSYahei.ttf'
    myfont = FontProperties(fname=fname)

    # 绘图
    plt.figure(figsize=(8, 8))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 5)
    plt.xlabel('TensoFlow Lite 识别',fontproperties=myfont)
    plt.ylabel('实际诊断',fontproperties=myfont)
    plt.text(0, 5, '0:N,正常搏动\n1:A,房性早搏\n2:V,室性早搏\n3:L,左束支传导阻滞\n4:R,右束支传导阻滞',fontproperties=myfont)
    plt.savefig("imx8mp_result.png")


def main():
    # X_train,Y_train为所有的数据集和标签集
    # X_test,Y_test为拆分的测试集和标签集
    X_train, Y_train, X_test, Y_test = loadData()
    Y_pred_class = []
    # plt.plot(np.float32(X_test)[0: 500, 0])
    # plt.savefig("mygraph.png")
    # if os.path.exists(model_path):
        # 导入训练好的模型
    # model = tf.keras.models.load_model(filepath=model_path)
    # else:
    #     # 构建CNN模型
    #     model = buildModel()
    #     model.compile(optimizer='adam',
    #                   loss='sparse_categorical_crossentropy',
    #                   metrics=['accuracy'])
    #     model.summary()
    #     # 定义TensorBoard对象
    #     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #     # 训练与验证
    #     model.fit(X_train, Y_train, epochs=30,
    #               batch_size=128,
    #               validation_split=RATIO,
    #               callbacks=[tensorboard_callback])
    #     model.save(filepath=model_path)

    # 预测
    # Y_pred = model.predict_classes(X_test)
    interpreter = tflite.Interpreter(model_path=test_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print("mode shape:",input_details[0]['shape'])
    # print(X_test.shape)
    print("Total", '%d' "signlas" % X_test.shape[0])
    # print(X_test[count,:,:].shape)
    for count in range(X_test.shape[0]):
        interpreter.set_tensor(input_details[0]['index'], np.float32(X_test[count,:,:].reshape(1,300,1)))
        if count < 10:
            startTime = time.time()
            interpreter.invoke()
            delta = time.time() - startTime
            print("First 10 inferences' time:", '%.1f' % (delta * 1000), "ms\n")
        else:
            interpreter.invoke()
        Y_pred = interpreter.get_tensor(output_details[0]['index'])
        #print("Y_pred result :", np.argmax(Y_pred))
        Y_pred_class.append(np.argmax(Y_pred))
        count += 1
    Y_pred_class = np.array(Y_pred_class)
    
    # 绘制混淆矩阵
    plotHeatMap(Y_test, Y_pred_class)


if __name__ == '__main__':
    main()
