import  numpy as np
from  LoadData import  loaddata
from SMO import smo
from  PredictFunction import predict_function
from matplotlib import  pyplot as plt

if __name__ == '__main__':

    train_path = 'traindata.txt'
    test_path  =  'testdata.txt'

    #读取数据，并且用矩阵的形式存储
    train_x, train_y, test_x, test_y = loaddata(train_path, test_path)
    train_x  = np.matrix(train_x)
    train_y = np.matrix(train_y).T
    test_x  = np.matrix(test_x)
    test_y = np.matrix(test_y).T    #y的标签都用列向量存储

    kenerl_dict = {1: 'gaussian', 2: 'linear'}

    '''s = input("请输入最大迭代次数maxk, 参数c(以空格隔开)：\n")

    n = int(input("核函数选择：输入1选高斯核函数，输入2选择线性核函数:\n"))

    kenerlopt = kenerl_dict[n]
    maxk = int(s.split(' ')[0])
    c = float(s.split(' ')[1])'''
    maxk = 50
    c = 5
    kenerlopt = kenerl_dict[1]
    e = 0.001

    Attributes = smo(train_x, train_y, test_x, test_y ,maxk, e, c, kenerlopt)


    #计算测试数据的准确度
    Accuracy = predict_function(Attributes)
    print("The test Accuray:",Accuracy)


    #画出训练数据的图
    p1 = plt.subplot(1,2,1)
    for i in range(0, Attributes.numTrainSamples):
        if Attributes.train_y[i] == 1:
            p1.scatter(Attributes.train_x[i, 0], Attributes.train_x[i, 1], c='r')
        else:
            p1.scatter(Attributes.train_x[i, 0], Attributes.train_x[i, 1], c = 'b')
    p1.set_title('The distribution of train_data', fontsize = 12)

    #画出测试数据图
    p2 = plt.subplot(1,2,2)
    for i in range(0, Attributes.numTestSamples):
        if Attributes.test_y[i] == 1:
            p2.scatter(Attributes.test_x[i, 0], Attributes.test_x[i, 1], c='r')
        else:
            p2.scatter(Attributes.test_x[i, 0], Attributes.test_x[i, 1], c = 'b')
    p2.set_title('The distribution of test_data', fontsize=12)

    plt.savefig('data_distribution.png', dpi = 600)

    plt.show()

