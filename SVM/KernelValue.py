#由于后续计算中会频繁用到核函数的值，故将所有的核函数值一次计算，并存在矩阵中

import numpy as np
from math import exp

def calculte_kenerl_value(Attributes, sample_x, s):

    kernel_type = Attributes.kernelOpt
    # s = 1表示计算的是训练数据的核函数值
    #s = 0 表示计算的是测试数据的核函数值
    dataset_dict = {0 : Attributes.test_x, 1 : Attributes.train_x}
    numSamples_dict = {0: Attributes.numTestSamples, 1 : Attributes.numTrainSamples}

    dataset_x = dataset_dict[s]
    numSamples = numSamples_dict[s]

    if kernel_type == 'linear':   #线性核函数

        kernel_value = dataset_x * sample_x.T + 1

    elif kernel_type == 'gaussian': #高斯核函数

        sigma = 1.0
        tmp_matrix = dataset_x - sample_x   #计算公式中 x-z 部分

        kernel_value = np.zeros((numSamples, 1) ) #初始化一个以训练样本数为维度的列向量，数值为0


        for i in range(0,numSamples):
            exponent = tmp_matrix[i, :] * tmp_matrix[i, :].T   #计算指数部分
            kernel_value[i,0] = exp(- exponent / (2 * sigma*sigma))   #表示第i个数据与sample_x的核函数值


    else:
        raise NameError('Not support kernel type!')

    return kernel_value