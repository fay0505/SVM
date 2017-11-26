#选择第二个α_j
import  numpy as np
from math import fabs
from CalculateError import cal_error


def select_j(Attributes, index_i, E_i):

    # 为寻找|E_i - E_j|最大，故先将所以数据的预测误差计算出
    Error_Matrix = np.zeros((Attributes.numTrainSamples, 1))


    for i in range(0, Attributes.numTrainSamples):
        Error_Matrix[i, 0] = cal_error(Attributes, i)

    MaxStep = 0
    a_j = 0
    E_j = 0
    index_j = 0

    for j in range(0, Attributes.numTrainSamples):

        tmp_step = fabs(Error_Matrix[j, 0] - E_i)  # 计算|E_i - E_j|
        if tmp_step > MaxStep:
            a_j = Attributes.a[j, :]
            MaxStep = tmp_step
            E_j = Error_Matrix[j, 0]
            index_j = j  # 记录αj的下标

    return index_i, E_j