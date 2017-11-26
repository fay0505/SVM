#计算预测的误差
from KernelValue import  calculte_kenerl_value

def cal_error(Attributes, index):

    sample_x = Attributes.train_x[index ,:]

    kernel_value = calculte_kenerl_value(Attributes, sample_x, 1)  #列矩阵
    Attributes.KernelMatrix[index, :] = kernel_value.T

    sum = 0
    for j in range(0, Attributes.numTrainSamples):

        sum += Attributes.train_y[j, 0] * Attributes.a[j, 0] * kernel_value[j, 0]      #这三个向量都是列向量

    Error_i = sum - Attributes.b - Attributes.train_y[index, 0]

    return Error_i
