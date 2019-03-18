# coding=utf-8
'''
Created on 2018-05-28 22:22:42
 
@author: FengLu
'''
from GenerateData import *


def GetU0(prior_rate):
    '''
    根据PricePref.txt文件，根据需要加入的先验知识的比例，获得U0和Gu
    '''
    PricePrefAddr = Addr + 'PricePref.txt'
    U0 = mat(ones((m, k))) * 0.01
    Gu = mat(zeros((m, m)))
    # chose_indexs = rd.sample(range(m),int(m*prior_rate))
    chose_indexs = range(m)[:int(m * prior_rate)]
    line_num = 0
    with open(PricePrefAddr, 'r')  as f:
        while True:
            line = f.readline()
            if line != '':
                if line_num in chose_indexs:
                    data = map(float, line.strip('\n').split('\t'))
                    U0[line_num,] = array(data)
                    Gu[line_num, line_num] = 1.0
                else:
                    pass
                line_num = line_num + 1
            else:
                break
    return U0, Gu, chose_indexs


def GetV0(prior_rate):
    '''
    根据category.txt以及先验知识比例，获得V0和Gv
    '''
    CategoryAddr = Addr + 'category.txt'
    category = array([])
    with open(CategoryAddr, 'r') as f:
        for line in f.readlines():
            category = append(category, int(line.strip('\n')))
    category_index = 0
    RealV = mat(zeros((n, k)))
    V0 = mat(ones((n, k))) * 0.01
    Gv = mat(zeros((n, n)))
    for i in range(n):
        start_index = sum(category[:category_index])
        middle_index = sum(category[:category_index]) + category[category_index] / 2
        end_index = sum(category[:category_index]) + category[category_index]
        if start_index <= i < middle_index:
            RealV[i, 0] = 1.0
        elif middle_index <= i < end_index:
            RealV[i, 1] = 1.0
        else:
            category_index = category_index + 1
            RealV[i, 0] = 1.0
    # chose_indexs = rd.sample(range(n),int(n*prior_rate))
    chose_indexs = range(m)[:int(n * prior_rate)]
    for j in chose_indexs:
        V0[j,] = RealV[j,]
        Gv[j, j] = 1.0
    return V0, Gv, chose_indexs


def LoadX():
    XAddr = Addr + 'X.txt'
    X = mat(ones((m, n))) * 0.01
    line_num = 0
    with open(XAddr, 'r') as f:
        while True:
            line = f.readline()
            if line != '':
                if line != '\n':
                    data = map(lambda x: map(float, x.split('|')), line.strip('\n').split('\t'))
                    for i in range(len(data)):
                        X[line_num, int(data[i][0])] = data[i][1]
                else:
                    pass
                line_num = line_num + 1
            else:
                break
    return X


def LoadULapras(UserFriends_num, PriceFriend_rate):
    FolderAddr = Addr + 'k' + str(UserFriends_num) + '_rate' + str(PriceFriend_rate) + '/'
    WAddr = FolderAddr + 'Wu_k' + str(UserFriends_num) + '_rate' + str(PriceFriend_rate) + '.txt'
    Wu = mat(zeros((m, m)))
    Du = mat(zeros((m, m)))
    line_num = 0
    with open(WAddr, 'r') as f:
        while True:
            line = f.readline()
            if line != '':
                data = map(int, line.strip('\n').split('\t'))
                for i in data:
                    Wu[line_num, i] = 1.0
                Du[line_num, line_num] = UserFriends_num
                line_num = line_num + 1
            else:
                break
    return Wu, Du


def LoadVLapras():
    WAddr = Addr + 'Wv_k' + str(ItemFriends_num) + '.txt'
    line_num = 0
    Wv = mat(zeros((n, n)))
    Dv = mat(zeros((n, n)))
    with open(WAddr, 'r') as f:
        while True:
            line = f.readline()
            if line != '':
                data = map(int, line.strip('\n').split('\t'))
                for i in range(len(data)):
                    Wv[line_num, data[i]] = 1.0
                Dv[line_num, line_num] = ItemFriends_num
                line_num = line_num + 1
            else:
                break
    return Wv, Dv


def ini(x, y):
    Init = mat(zeros((x, y)))
    for i in range(x):
        for j in range(y):
            Init[i, j] = rd.uniform(0, 1)
    return Init


def norm(a):
    x, y = shape(a)
    sum_list = a.sum(axis=1)
    for i in range(x):
        a[i,] = a[i,] / sum_list[i,]
    return a


def update_U(U, H, V, Wu, Du):
    me = X * V * H.T + Alpha_U * Gu * U0 + Alpha_Lu * Wu * U
    de = U * H * V.T * V * H.T + Alpha_U * Gu * U + Alpha_Lu * Du * U
    new_U = mat(ones((m, k))) * 0.01
    for i in range(m):
        for j in range(k):
            new_U[i, j] = U[i, j] * sqrt(me[i, j] / de[i, j])
    return new_U


def update_V(U, H, V, Wv, Dv):
    me = X.T * U * H + Alpha_V * Gv * V0 + Alpha_Lv * Wv * V
    de = V * H.T * U.T * U * H + Alpha_V * Gv * V + Alpha_Lv * Dv * V
    new_V = mat(ones((n, k))) * 0.01
    for i in range(n):
        for j in range(k):
            new_V[i, j] = V[i, j] * sqrt(me[i, j] / de[i, j])
    return new_V


def update_H(U, H, V):
    me = U.T * X * V
    de = U.T * U * H * V.T * V
    new_H = mat(ones((k, k))) * 0.01
    for i in range(k):
        for j in range(k):
            new_H[i, j] = H[i, j] * sqrt(me[i, j] / de[i, j])
    return new_H


def RunModel(ModelPara, IterAddr):
    Wu, Du, Wv, Dv = ModelPara
    U = U0;
    V = V0
    H = (U.T * U).I * (U.T * X * V) * (V.T * V).I
    SaveEachUAddr = IterAddr + 'eachU.txt'
    SaveEachVAddr = IterAddr + 'eachV.txt'
    SaveEachHAddr = IterAddr + 'eachH.txt'
    PartAddr = IterAddr + 'part.txt'
    with open(SaveEachUAddr, 'w') as f1, open(SaveEachVAddr, 'w') as f2, open(SaveEachHAddr, 'w') as f3, open(PartAddr,
                                                                                                              'w') as f4:
        for i in range(Iter):
            if i % 100 == 0:
                print(i)
            # Update
            H = update_H(U, H, V)
            U = update_U(U, H, V, Wu, Du)
            V = update_V(U, H, V, Wv, Dv)
            # U = norm(new_U); V = norm(new_V); H = norm(new_H)

            # Save
            f1.write(','.join(map(lambda x: '|'.join(map(str, x)), U.tolist())) + '\n')
            f2.write(','.join(map(lambda x: '|'.join(map(str, x)), V.tolist())) + '\n')
            f3.write(','.join(map(lambda x: '|'.join(map(str, x)), H.tolist())) + '\n')

            # Statistics
            Part1 = X - U * H * V.T
            Part2 = Gu * (U - U0)
            Part3 = U.T * (Du - Wu) * U
            Part4 = Gv * (V - V0)
            Part5 = V.T * (Dv - Wv) * V
            sta1 = linalg.norm(Part1)
            sta2 = Alpha_U * linalg.norm(Part2)
            sta3 = Alpha_Lu * trace(Part3)
            sta4 = Alpha_V * linalg.norm(Part4)
            sta5 = Alpha_Lv * trace(Part5)
            writeline = ','.join(map(str, array([sta1, sta2, sta3, sta4, sta5]))) + '\n'
            f4.write(writeline)
    U = norm(U);
    V = norm(V)
    return U, V


def SaveModel(FriendnumRange, RateRange):
    for UserFriends_num in FriendnumRange:
        for PriceFriend_rate in RateRange:
            print(UserFriends_num, PriceFriend_rate)
            SaveAddr = Addr + 'k' + str(UserFriends_num) + '_rate' + str(PriceFriend_rate) + '/'
            IterAddr = SaveAddr + str(Iter) + '/'
            try:
                os.makedirs(IterAddr)
            except:
                pass
            SaveUAddr = IterAddr + 'U.txt'
            SaveVAddr = IterAddr + 'V.txt'
            f1 = open(SaveUAddr, 'w')
            f2 = open(SaveVAddr, 'w')

            Wu, Du = LoadULapras(UserFriends_num, PriceFriend_rate)
            Wv, Dv = LoadVLapras()
            ModelPara = [Wu, Du, Wv, Dv]
            # Run
            print('Start...')
            U, V = RunModel(ModelPara, IterAddr)
            f1.write('\n'.join(map(lambda x: ','.join(map(str, x)), U.tolist())))
            f2.write('\n'.join(map(lambda x: ','.join(map(str, x)), V.tolist())))

            f1.close();
            f2.close()


def writein(data, name):
    with open(Addr + name + '.txt', 'w') as f:
        f.write('\n'.join(map(str, data)))


if __name__ == '__main__':
    # generated-data parameters
    k = 2  # 类别数为2，即低价值和高价值的偏好
    DataPara = [2000, 1000, 20, 10, 0.1, 1, 3, 1, 100]
    m, n, c, category_var, buyitem_rate, buyitem_var, buynum_mean, buynum_var, ItemFriends_num = DataPara
    timestr = '1547532289.74'  # 生成数据对应的时间
    Addr = './data-' + timestr + '/'

    # model setting
    UserPrior_rate = 0.6;
    ItemPrior_rate = 0.6;
    Iter = 1000
    FriendnumRange = arange(20, 210, 10)
    RateRange = arange(0.0, 1.1, 0.1)

    Alpha_U, Alpha_V, Alpha_Lu, Alpha_Lv = ones((4,))

    # load data
    U0, Gu, chose_users = GetU0(UserPrior_rate)
    V0, Gv, chose_items = GetV0(ItemPrior_rate)
    writein(chose_users, 'chose_users')
    writein(chose_items, 'chose_items')
    X = LoadX()

    # run&save model
    SaveModel(FriendnumRange, RateRange)
