#构造决策函数，预测数据的标签y

from KernelValue import  calculte_kenerl_value

def predict_function(Attributes):

    predict_y = []  #存储预测标签

    test_x = Attributes.test_x
    test_y = Attributes.test_y
    num = test_x.shape[0]  # 测试数据的个数

    for i in range(0, num):
        kernel_value = calculte_kenerl_value(Attributes, test_x[i,:], 0)
        sum = 0
        for j in range(0 ,num):
            sum += Attributes.a[j] * test_y[j] * kernel_value[j, 0]
        sum -= Attributes.b

        if sum >= 0 :
            predict_y.append(1)
        else:
            predict_y.append(-1)

    correct_num = 0   #计算预测的准确数
    for m in range(0, len(predict_y)):
        if predict_y[m] == test_y[m]:
            correct_num += 1

    accuracy = correct_num / len(test_y)

    return accuracy