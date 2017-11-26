import numpy as np

from  InnerLoop import innerloop

class SVM_Struct:
    def __init__(self, train_x, train_y, test_x, test_y, c, e, kerneloption):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.c = c
        self.e = e   #精度
        self.numTrainSamples = self.train_x.shape[0]  #训练样本的数量
        self.numTestSamples = self.test_x.shape[0]
        self.a = np.zeros((self.numTrainSamples, 1))
        self.b = 0
        self.kernelOpt = kerneloption
        self.ErrorMatrix = np.zeros((self.numTrainSamples, 1))#样本的预测误差矩阵
        self.KernelMatrix = np.zeros((self.numTrainSamples, self.numTrainSamples))


def smo(train_x, train_y, test_x, test_y ,maxk, e, c, kerneloption):       #a表示α，maxk表示迭代最大次数， e表示精度

    Attributes = SVM_Struct(train_x, train_y,test_x, test_y, c, e, kerneloption)
    k = 0     #记录迭代次数

    pairschanged = 0   #用来记录优化过的α对数
    entire_train_set = True     #标识是否需要遍历整个训练集


    #进行外层循环
    while k < maxk and (pairschanged > 0 or entire_train_set):

        pairschanged = 0
        if entire_train_set:
            for i in range(0, Attributes.numTrainSamples):

                    pairschanged += innerloop(Attributes,i)
            print("Iterate entire trian set , K = %d ,pairschanged = %d" % (k, pairschanged))
            k += 1

        else:
            for i in range(0, Attributes.numTrainSamples):

                if Attributes.a[i] > 0 and Attributes.a[i] < Attributes.c:    #遍历满足0< a < c的所有样本

                    pairschanged += innerloop(Attributes, i)

            print("Iterate non-bound  subset , K = %d ,pairschanged = %d" % (k, pairschanged))

            k += 1

        if entire_train_set:
            entire_train_set = False

        elif pairschanged == 0:
            entire_train_set = True

    return Attributes

