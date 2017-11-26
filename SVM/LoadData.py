#对数据进行预处理，将标签和X分开


def loaddata(trainpath, testpath):

    train_x = []
    train_y = []
    test_x  = []
    test_y = []



    with open(trainpath, 'r') as f1:
        lines = f1.readlines()
        for eachline in lines:
            tmp = eachline.split(',')

            train_x.append((float(tmp[0]), float(tmp[1]) ))    #每一个样本的两个x值以元组形式插入到列表
            train_y.append(float(tmp[-1].split('\n')[0]))

    f1.close()

    with open(testpath, 'r') as f2:
        lines = f2.readlines()
        for eachline in lines:
            tmp = eachline.split(',')
            test_x.append((float(tmp[0]), float(tmp[1])))
            test_y.append(float(tmp[-1].split('\n')[0]))

    f2.close()

    return train_x, train_y, test_x, test_y