#对所有训练样本进行外层循环，

import numpy as np
from random import randint
from CalculateError import cal_error
from Select_j import select_j
from TakeStep import takestep


def innerloop(Attributes,index_i):    #y_i代表

    a_i = Attributes.a[index_i, 0]
    y_i = Attributes.train_y[index_i, 0]
    E_i = cal_error(Attributes, index_i)

    #寻找到所有满足0 < α < C的α下标
    non_bound_a = []
    for m in range(0, Attributes.numTrainSamples):
        if Attributes.a[m, 0] < Attributes.c and Attributes.a[m, 0] > 0:
            non_bound_a.append([m, 0])   #存储满足条件的α，0表示未被选中

    num_non_bound = len(non_bound_a)     #计算满足条件的α个数


    #判断是否违反KKT条件
    if((E_i * y_i < - Attributes.e and a_i < Attributes.c) or (E_i * y_i > Attributes.e and a_i > 0)):


        if num_non_bound > 1:

            index_j, E_j = select_j(Attributes, index_i, E_i)
            if takestep(Attributes, index_i, E_i, index_j, E_j):
                return 1

        #上述方法不能使目标函数有足够下降，采用以下启发式规则继续选择αj

        #以一个随机的起始点开始，遍历集合 0 < α < C，即在间隔边界上的支持向量点，依次试用
        flag = 0  #计算集合 0 < α < C 已经被遍历过的α 个数
        while flag < num_non_bound:
            while 1 :
                index = randint(0, num_non_bound - 1)

                if non_bound_a[index][1] == 0:

                    non_bound_a[index][1] == 1
                    flag += 1
                    break
            index_j = non_bound_a[index][0]
            E_j = cal_error(Attributes, index_j)

            if takestep(Attributes, index_i, E_i, index_j, E_j):
                return 1


        #若上面的方法还是找不到适合的αj，遍历整个数据集
        flag_matrix = np.zeros((Attributes.numTrainSamples, 1))  #用来记录是否已经遍历过α
        flag1 = 0
        while flag1 < Attributes.numTrainSamples:
            tmp_index = randint(0, Attributes.numTrainSamples - 1)
            if flag_matrix[tmp_index, 0] == 0:    #未被遍历

                flag_matrix[tmp_index, 0] == 1
                flag1 += 1
            index_j = tmp_index
            E_j = cal_error(Attributes, index_j)

            if takestep(Attributes, index_i, E_i, index_j, E_j):
                return 1

    return 0






