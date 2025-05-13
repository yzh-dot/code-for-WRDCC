# 数据驱动不确定集合需要根据以下步骤
# 1.获取风电数据
# 2.调用get_D_para获取D
# 3.根据D，调用get_epsilon_para获取epsilon（模糊集半径）
# 4.根据epsilon调用get_uncertainty_set获取不确定集边长L


import numpy as np

# 1.1 首先获取D，lb和ub根据情况设置，default = 0,10000
def get_D_para(error_data, lb, ub):
    D = p_bisearch(error_data, lb, ub)
    return D

# 1.2 求解D之前需要根据二分法确定p的取值，a和b为get_D_para中初始lb和ub，default = 0,10000
def p_bisearch(error_data,a,b):
    lb = a + 0.382*(b-a)
    ub = a + 0.618*(b-a)
    iter = 1
    max_iter = 100000
    tol_gap = b-a
    while (tol_gap > 1e-5 and iter < max_iter):
        # 代入p的lb和ub来求取相应的D的值
        flb = compute_D_function(lb,error_data)
        fub = compute_D_function(ub,error_data)
        if flb > fub:
            a = lb
            lb = ub
            ub = a + 0.618 * (b-a)
        else:
            b = ub
            ub = lb
            lb = a + 0.382*(b-a)
        iter = iter + 1
        tol_gap = abs(b-a)
    if iter == max_iter:
        print("迭代次数已达到maxiter,未找到最小值")
    return compute_D_function((a+b)/2,error_data)


# def p_bisearch(error_data,a,b):
#     lb = a
#     ub = b
#     iter = 1
#     max_iter = 100000
#     tol_gap = b-a
#     while (tol_gap > 1e-5 and iter < max_iter):
#         # 代入p的lb和ub来求取相应的D的值
#         flb = compute_D_function(lb,error_data)
#         fub = compute_D_function(ub,error_data)
#         if flb > fub:
#             a = lb
#             lb = ub
#             ub = a + 0.618 * (b-a)
#         else:
#             b = ub
#             ub = lb
#             lb = a + 0.382*(b-a)
#         iter = iter + 1
#         tol_gap = abs(b-a)
#     if iter == max_iter:
#         print("迭代次数已达到maxiter,未找到最小值")
#     print(tol_gap)
#     print(iter)
#     return compute_D_function((a+b)/2,error_data)


# 1.3 计算D的值，依据文献Distributionally Robust Chance-Constrained Approximate AC-OPF With Wasserstein Metric公式（25）
def compute_D_function(p, error_data):
    # 计算数据的平均数
    mean = np.mean(error_data)
    # 对数据加和
    #sum_exp = np.sum(np.exp(p * (error_data - mean)**2))
    # 截断方式，防止溢出
    #clipped_data = np.clip(p * (error_data - mean)**2, -100, 100)  # exp函数的输入值最好不超过100
    data = p * (error_data - mean)**2
    # 对数据加和
    #sum_exp = np.sum(np.exp(clipped_data))
    sum_exp = np.sum(np.exp(data))
    # 最终的计算结果
    D = 2 * np.sqrt((1 + np.log(sum_exp / len(error_data))) / (2 * p))
    return D

# 2.1 获取epsilon，lb和ub根据情况设置，default = 0,10000，beta为置信度，default = 0.95
def get_epsilon_para(error_data, lb, ub, beta):
    N = len(error_data)
    # 可提前计算D的值，也可重新计算D的值
    D = get_D_para(error_data, lb, ub)
    # 依据Wasserstein Metric Based Distributionally Robust Approximate Framework for Unit Commitment公式（4）
    epsilon = D * np.sqrt((2 * np.log(1 / (1 - beta))) / N)
    return epsilon


# 3.1 获取数据驱动的不确定集，依据get_epsilon_para获得epsilon，default beta = 0.95
def get_uncertainty_set(error_data, beta, epsilon):
    # 对初始数据标准化
    stand_data, mean, var = data_standardized(error_data)
    # 获取L
    L = getLpara(stand_data, beta, epsilon)
    uncertainty_set_data = []
    for i in range(len(stand_data)):
        if abs(stand_data[i]) < L:
            item = stand_data[i] * np.sqrt(var) + mean
            uncertainty_set_data.append(item)
    L_array = np.array(uncertainty_set_data)
    upperOmega = np.max(L_array)
    lowerOmega = np.min(L_array)
    print("upperOmega is",upperOmega)
    print("lowerOmega is",lowerOmega)
    return L_array

# 3.2 数据标准化
def data_standardized(error_data):
    mean = np.mean(error_data)
    # 计算数据的方差
    var = np.var(error_data)
    # 对数据进行标准化
    stand_data = (error_data - mean) / np.sqrt(var)
    return stand_data, mean, var

# 3.3 获取L的值，依据get_epsilon_para获得epsilon，default beta = 0.95
# 理论根据Distributionally Robust Chance-Constrained Approximate AC-OPF With Wasserstein Metric中算法1：Nested Bisection Search
def getLpara(stand_data,beta,epsilon):
    lb = 0
    ub = 100
    while(ub-lb>1e-4):
        L = (lb+ub)/2
        # 计算固定L下的函数值,对k进行搜索
        gamma = k_bisearch(stand_data,0,100, epsilon, L)
        if (gamma > beta):
            lb = L
        else:
            ub = L
    return (lb+ub)/2

# 3.4 在固定不确定集边界L下，根据二分法获取k的值，a和b分别为lb,ub:0,100
def k_bisearch(stand_data,a,b,epsilon,L):
    lb = a + 0.382*(b-a)
    ub = a + 0.618*(b-a)
    iter = 1
    max_iter = 100000
    tol_gap = b-a
    h_all = np.zeros(2*max_iter)
    while (tol_gap > 1e-5 and iter < max_iter):
        flb = h_function(L,lb,epsilon,stand_data)
        fub = h_function(L,ub,epsilon,stand_data)
        h_all[2*(iter-1)] = flb
        h_all[2*iter-1] = fub
        if flb > fub:
            a = lb
            lb = ub
            ub = a + 0.618 * (b-a)
        else:
            b = ub
            ub = lb
            lb = a + 0.382*(b-a)
        iter = iter + 1
        tol_gap = abs(b-a)
    if iter == max_iter:
        print("迭代次数已达到maxiter,未找到最小值")
    # print(min(h_all[0:2*iter-2]))
    # print(h_function(L,(a+b)/2,epsilon,stand_data))
    return h_function(L,(a+b)/2,epsilon,stand_data)


# 3.5 根据Wasserstein Metric Based Distributionally Robust Approximate Framework for Unit Commitment公式（10）
def h_function(l,k,epsilon,stand_data):
    sum = 0
    for i in range(len(stand_data)):
        midsum = k*max(l-abs(stand_data[i]),0)
        sum = sum + max(1-midsum,0)
    return k*epsilon + sum/len(stand_data)

# import pandas as pd
# n = 100000
# np.random.seed(41)
# #winddata = 3*np.random.randn(n)
# data_all = pd.read_excel('windErrorTenYearsData.xlsx')
# wind_f = data_all['WindForecast'].values
# wind_t = data_all['windTrue'].values
# winddata = wind_f-wind_t
# #winddata= np.random.randint(5, 500, n)
# D = get_D_para(winddata,0,10000)
# epsilon = get_epsilon_para(winddata,0,10000,0.95)
# L = get_uncertainty_set(winddata, 0.95, epsilon)
# print("epsilon is",epsilon)


