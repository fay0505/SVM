from copy import deepcopy
from  math import  fabs

def takestep(Attributes, index_i, E_i, index_j, E_j):

    if index_i == index_j :      #如果内层和外层循环选择的是同一个α，则返回重选
        return 0


    a_i = Attributes.a[index_i, 0]
    a_j = Attributes.a[index_j, 0]

    train_x = Attributes.train_x
    train_y = Attributes.train_y

    #计算L和H
    if train_y[index_j, 0] != train_x[index_i, 0]:

        L = max(0, a_j - a_i)
        H = min(Attributes.c, Attributes.c + a_j - a_i)
    else:

        L = max(0, a_j + a_i - Attributes.c)
        H = min(Attributes.c, a_j + a_i)

    if L == H:
        return 0

    #找到αi αj对应的标签
    yi = Attributes.train_y[index_i, 0]
    yj = Attributes.train_y[index_j, 0]
    #计算η
    Kii  = Attributes.KernelMatrix[index_i, index_i]
    Kjj = Attributes.KernelMatrix[index_j, index_j]
    Kij = Attributes.KernelMatrix[index_i, index_j]

    Eta = Kii + Kjj - 2 * Kij

    s = yi * yj
    #根据Eta的情况计算剪辑后的αj
    if (Eta > 0):
        aj_unc = a_j + yj * (E_i - E_j) / Eta
        # 计算剪辑后的a_j
        if aj_unc >= H:
            aj_new = H
        elif aj_unc <= L:
            aj_new = L
        else:
            aj_new = deepcopy(aj_unc)
    else:
        f1 = yi * (E_i * Attributes.b) - a_i * Kii - s * a_j * Kij
        f2 = yj * (E_j * Attributes.b) - s * a_i * Kij - a_j * Kjj
        L1 = a_i + s * (a_j - L)
        H1 = a_i + s * (a_j - H)
        FunL = L1 * f1 + L * f2 + 0.5 * (L1**2) * Kii + 0.5 * (L**2) * Kjj + s * L * L1 * Kij
        FunH = H1 * f1 + H * f2 + 0.5 * (H1**2) * Kii + 0.5 * (H**2) * Kjj + s * H1 * H * Kij
        if FunL < FunH - Attributes.e:
            aj_new = L
        elif FunL > FunH + Attributes.e:
            aj_new = H
        else:
            aj_new = deepcopy(a_j)


    if fabs(a_j - aj_new) < Attributes.e * (a_j + aj_new + Attributes.e):
        return 0

    ai_new = a_i + s * (a_j - aj_new)

    # 计算b
    bi = E_i + yi * (ai_new - a_i) * Kii + yj * (aj_new - a_j) * Kij + Attributes.b
    bj = E_j + yi * (ai_new - a_i) * Kij + yj * (aj_new - a_j) * Kjj + Attributes.b

    if (ai_new < Attributes.c and ai_new > 0) and (aj_new < Attributes.c and aj_new > 0):
        Attributes.b = bi
    else:
        Attributes.b = (bi + bj) / 2

    Attributes.a[index_i, 0] = ai_new
    Attributes.a[index_j, 0] = aj_new

    return 1

