import pandas as pd
import matplotlib.pyplot as plt
from rsome import grb_solver as grb
from rsome import cpt_solver as cpt
from rsome import dro
from rsome import ro
from rsome import E
import rsome as rso
import time
import random
import os
import AmbiguitySet
import seaborn as sns
import numpy as np
from scipy.stats import truncnorm
from WDRO_118_Bus import WDRO_118
np.set_printoptions(threshold=np.inf)
## 1.读取数据

def WDRCC_118_Bus(model,zeta,epsilon):
    units = pd.read_excel('data\IEEE 118 bus.xls', 'Generator')     # Parameters of generators
    branch = pd.read_excel('data\IEEE 118 bus.xls', 'Branch')       # Parameters of branch
    Load = pd.read_excel('data\IEEE 118 bus.xls', 'Load')           # Paremeters of Load

    a = units['a ($/MW^2)'].values
    b = units['b ($/MW)'].values
    c = units['c ($)'].values
    g_max = units['Maxmum Output (MW)'].values
    g_min = units['Minimum Output (MW)'].values
    g_init= units['Initial Output'].values
    r_up = units['Ramping Up (MW/h)'].values
    r_down = units['Ramping Down (MW/h)'].values
    up_sr_max = units['Maximum up reserve capacity (MW)'].values
    down_sr_max = units['Maximum down reserve capacity (MW)'].values
    C_up_sr = units['Upward reserve capacity cost ($/MWh)'].values
    C_down_sr = units['Downward reserve capacity cost ($/MWh)'].values
    start_cost = units['Start-up cost ($)'].values
    down_cost = units['Shut-down Cost ($)'].values
    h_up = units['min_on (h)'].values
    h_down = units['min_off (h)'].values
    capacity = branch['Capacity'].values
    u_init = (g_init > 0).astype(int)
    reactance = branch['X'].values
    load = Load['load_all'].values[0:24]
    load_rate = Load['Load Ratio'].values
    load_bus = Load['Load Bus'].values
    lines = len(reactance)            # 支路数
    numnodes = 118                    # 节点数

    np.random.seed(0)
    ## 3.model基本参数
    S = 5                       # number of scenarios
    T = 24                      # hours
    N = units.shape[0]          # Number of units
    ESS = 3                     # 储能数量
    W = 6                       # 风机数量
    N_sample = zeta.shape[0]    # 样本量
    model = ro.Model(S)         # robust model
    C_cut = np.random.randint(5, 10, W)        # 弃风惩罚代价系数

    ## 4. 创建dvar
    u = model.dvar((T, N), 'B')                      # Unit commitment statuses
    v = model.dvar((T, N), 'B')                      # Switch-on statuses of units
    w = model.dvar((T, N), 'B')                      # Switch-off statuses of units
    g = model.dvar((T, N))                           # Generation outputs
    p_w_cut = model.dvar((T,W))                      # wind power cut
    sum_power_GSDF = model.dvar((T,lines))           # 每个节点的所有机组注入输出功率和
    u_ch = model.dvar((T,ESS), 'B')                  # 储能充电状态
    u_dis = model.dvar((T,ESS), 'B')                 # 储能放点状态
    p_ch = model.dvar((T,ESS))                       # 储能充电量
    p_dis = model.dvar((T,ESS))                      # 储能放电量
    S_ess = model.dvar((T+1,ESS))                      # 储能量
    lambda_ = model.dvar()                           # 拉格朗日变量
    alpha_g = model.dvar((T,N))                      # 燃料机组调整参数
    alpha_w = model.dvar((T,W))                      # 风电机组调整参数
    alpha_dis = model.dvar((T,ESS))                  # 放电调整参数
    alpha_ch = model.dvar((T,ESS))                   # 充电调整参数
    up_sr = model.dvar((T,N))                        # spinning up
    down_sr = model.dvar((T,N))                      # spinning down
    g_alpha_g = model.dvar((2*T*N))                  # 用于构建二次规划参数


    N_sample = zeta.shape[0]

    ## 5.创建cost func
    ## 原始求解(并没有保证Q是半正定矩阵)
    # Q = np.zeros((2*N*T,2*N*T))                 # 二次规划系数矩阵

    # for i in range(T*N,2*T*N):
    #     Q[i,i] = a[i%N]*(zeta*zeta).sum()/N_sample       # D2部分构建

    # for i in range(T*N):                        
    #     j = i+T*N
    #     Q[i,j] = -2*a[i%N]*zeta.sum()/N_sample           # D1部分构建

    # for i in range(T*N):
    #     Q[i,i] = a[i%N]/N_sample                         # D0部分构建


    ## 理论上求解（当Q为非半正定矩阵时需要对特征值进行处理）
    Q = np.zeros((2*N*T,2*N*T)) 

    for i in range(T*N,2*T*N):
        Q[i,i] = a[i%N]*(zeta*zeta).sum()/N_sample       # D2部分构建

    for i in range(T*N):                        
        j = i+T*N
        Q[i,j] = -a[i%N]*zeta.sum()/N_sample           # D1部分构建
        Q[j,i] = -a[i%N]*zeta.sum()/N_sample

    for i in range(T*N):
        Q[i,i] = a[i%N]/N_sample                         # D0部分构建
    eig_delta = 1e-8
    # 计算 Q 的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    # 将负的特征值设为 0，保证半正定
    if ~eigenvalues.all()>=0:
        eigenvalues[eigenvalues < 0] = eig_delta
        # 重构 Q，使其为半正定
        Q = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    
    D = rso.quad(g_alpha_g,Q)                   # 二次规划形式

    # objective optimization
    model.min((start_cost*v+down_cost*w).sum()+lambda_*epsilon + 
            D-1/N_sample*(((b*alpha_g).sum()+(alpha_w*C_cut).sum())*zeta.sum())+
            (b*g + c*u).sum()+(C_up_sr*up_sr+C_down_sr*down_sr).sum()+(C_cut*p_w_cut).sum()) 

    ## 6. 创建约束

    # 6.1 二次规划优化项约束，为保证参数一致

    # 前一半参数为机组发电
    for t in range(T):
        model.st(g_alpha_g[0+t*N:(t+1)*N] == g[t])

    # 后一半参数为机组调整系数
    for t in range(T,2*T):
        model.st(g_alpha_g[t*N:(t+1)*N] == alpha_g[t-T])

    # 拉格朗日变量
    model.st(lambda_>=0) 


    # 6.2 二次规划导数约束（为减少样本数量增加带来的复杂计算，针对构建的Phi函数的导数进行约束）

    ## 原始求解（并没有考虑Q1和Q2是否为半正定矩阵，且并不对称）

    # Q1 = np.zeros((2*N*T,2*N*T)) # 取样本最大值时二次规划约束的系数矩阵

    # # D2部分构建(取样本最大值)
    # for i in range(T*N,2*T*N):
    #     Q1[i,i] = 2*a[i%N]*max(zeta)

    # # D1部分构建
    # for i in range(T*N):
    #     j = i+T*N
    #     Q1[i,j] = -2*a[i%N]



    # Q2 = np.zeros((2*N*T,2*N*T)) # 取样本最小值时二次规划约束的系数矩阵

    # # D2部分构建(取样本最小值)
    # for i in range(T*N,2*T*N):
    #     Q2[i,i] = -2*a[i%N]*min(zeta)

    # # D1部分构建
    # for i in range(T*N):
    #     j = i+T*N
    #     Q2[i,j] = 2*a[i%N]

    ## 理论上求解（对于非半正定矩阵中的负特征值转化为正的）

    # D2部分构建(取样本最大值)
    Q1 = np.zeros((2*N*T,2*N*T))

    for i in range(T*N,2*T*N):
        Q1[i,i] = 2*a[i%N]*max(zeta)

    # D1部分构建
    for i in range(T*N):
        j = i+T*N
        Q1[i,j] = -a[i%N]
        Q1[j,i] = -a[i%N]


    # D2部分构建(取样本最小值)
    Q2 = np.zeros((2*N*T,2*N*T))

    for i in range(T*N,2*T*N):
        Q2[i,i] = -2*a[i%N]*min(zeta)

    # D1部分构建
    for i in range(T*N):
        j = i+T*N
        Q2[i,j] = a[i%N]
        Q2[j,i] = a[i%N]

    # 计算 Q 的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(Q1)
    # 将负的特征值设为 0，保证半正定
    eigenvalues[eigenvalues < 0] = eig_delta
    # 重构 Q，使其为半正定
    Q1 = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


    # 计算 Q 的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(Q2)
    # 将负的特征值设为 0，保证半正定
    eigenvalues[eigenvalues < 0] = eig_delta
    # 重构 Q，使其为半正定
    Q2 = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # 二次规划约束
    model.st(rso.quad(g_alpha_g,Q1)-(b*alpha_g).sum()-(alpha_w*C_cut).sum()<=lambda_)
    model.st(rso.quad(g_alpha_g,Q2)+(b*alpha_g).sum()+(alpha_w*C_cut).sum()<=lambda_)

    #model.st(lambda_<=500)

    # 6.3 储能约束
    # np.random.seed(0)
    p_ch_max = np.random.randint(0, 20, ESS)           # 充电最大值
    p_dis_max = np.random.randint(0, 20, ESS)          # 放电最大值
    S_ess_max = np.random.randint(50, 100, ESS)        # 储能最大值 
    eta_ch = np.random.uniform(0.7, 0.9, ESS)           # 充电效率
    eta_dis = np.random.uniform(0.7, 0.9, ESS)          # 放电效率

    p_ch_max = np.array([60,90,180])           # 充电最大值
    p_dis_max = np.array([60,90,180])          # 放电最大值
    S_ess_max = np.array([400,500,1000])        # 储能最大值 

    p_ch_max = p_ch_max[:ESS]
    p_dis_max = p_dis_max[:ESS]
    S_ess_max = S_ess_max[:ESS]

    eta_ch = np.array([0.9,0.9,0.9])
    eta_dis = np.array([0.9,0.9,0.9])


    #model.st(u_ch>=0,u_dis>=0)
    model.st(u_ch+u_dis<=1)                                                                     # 充放电状态只取其一
    model.st(0<=p_ch,p_ch<=u_ch*p_ch_max)                                                       # 充电约束
    model.st(0<=p_dis,p_dis<=u_dis*p_dis_max)                                                   # 放电约束
    model.st(0<=S_ess,S_ess<=S_ess_max)                                                         # 储能约束
    S_ess_initial = 0
    model.st(S_ess_initial == S_ess[T-1],S_ess[0] == S_ess_initial, S_ess[1:] == S_ess[:-1] + eta_ch*p_ch[:]-1/eta_dis*p_dis[:])   # 储能前后状态约束

    # 6.4 启停约束
    model.st(v[0] >= u[0] - u_init,
            v[1:] >= u[1:] - u[:-1],
            v >= 0)                                 # Switch-on statuses
    model.st(w[0] >=  u_init - u[0],
            w[1:] >= u[:-1] - u[1:],
            w >= 0)                                 # Switch-off statuses

    # 6.5 最小启停时间约束

    indices_min_on = np.where(h_up > 0)[0]
    indices_min_off = np.where(h_down > 0)[0]
    for n in indices_min_on:
        model.st(v[t-h_up[n]+1:t+1, n].sum() <= u[t, n]
                for t in range(h_up[n], T))         # Minimum up time constraints
    for n in indices_min_off:
        model.st(w[t-h_down[n]+1:t+1, n].sum() <= 1 - u[t, n]
                for t in range(h_down[n], T))       # Minimum down time constraints

    # 6.6 功率约束

    # 风电预测值
    # np.random.seed(0)
    p_w_f_max = 300
    p_w_f = np.random.normal(200, np.sqrt(1500), (T,W))
    p_w_f[p_w_f>=p_w_f_max] = p_w_f_max
    #p_w_f = 70


    # 功率平衡约束
    model.st(g.sum(axis=1)+(p_w_f-p_w_cut).sum(axis=1)+p_dis.sum(axis=1)-p_ch.sum(axis=1)==load)


    # 6.7 最大最小功率约束
    model.st(g >= u*g_min+down_sr,                           # Minimum capacities of units
            g <= u*g_max-up_sr,
            down_sr>=0,down_sr<=down_sr_max,
            up_sr>=0,up_sr<=up_sr_max)                           # Maximum capacities of units

    # 风电约束
    model.st(p_w_cut >= 0,
            p_w_cut<=p_w_f)


    # 6.8  上下爬坡约束
    model.st(g[0] - g_init <= r_up,
            (g[1:]+up_sr[1:])-(g[:-1]-down_sr[:-1])<= r_up*(1+u[:-1]-u[1:])+r_up*(2-u[:-1]-u[1:]),
            g_init - g[0] <= r_down,
            (g[:-1]+up_sr[:-1]) - (g[1:]-down_sr[1:]) <= r_down*(1-u[:-1]+u[1:])+r_down*(2-u[:-1]-u[1:]))


    # 6.9 chance constraints （基于文献A Linear Programming Approximation of
    #  Distributionally Robust Chance-Constrained Dispatch With Wasserstein Distance）

    chance_num = 6                  # 机会约束个数
    tau = 0.05                      # 置信度
    t1 = model.dvar(chance_num)     # 与epsilon相乘系数
    t2 = model.dvar(chance_num)     # 与tau相乘系数


    #（1）ramp up和ramp down约束
    model.st(t1[0]*epsilon-tau*t2[0]<=-1/N_sample*alpha_g*(zeta.sum())+(up_sr-t2[0]))
    model.st(t1[0]*epsilon-tau*t2[0]<=1/N_sample*alpha_g*(zeta.sum())+(down_sr-t2[0]))
    model.st(alpha_g<=t1[0])

    #（2）弃风约束
    model.st(t1[1]*epsilon-tau*t2[1]<=-1/N_sample*alpha_w*(zeta.sum())+(p_w_f-p_w_cut-t2[1]))
    model.st(t1[1]*epsilon-tau*t2[1]<=1/N_sample*alpha_w*(zeta.sum())+(p_w_cut-t2[1]))
    model.st(alpha_w<=t1[1])    

    #（3）机组发电约束
    model.st(t1[2]*epsilon-tau*t2[2]<=-1/N_sample*alpha_g*(zeta.sum())+(g-t2[2]))
    model.st(t1[2]*epsilon-tau*t2[2]<=1/N_sample*alpha_g*(zeta.sum())+(u*g_max-g-t2[2]))
    model.st(alpha_g<=t1[2])

    #（4）储能放电约束
    model.st(t1[3]*epsilon-tau*t2[3]<=-1/N_sample*alpha_dis*(zeta.sum())+(p_dis-t2[3]))
    model.st(t1[3]*epsilon-tau*t2[3]<=1/N_sample*alpha_dis*(zeta.sum())+(u_dis*p_dis_max-p_dis-t2[3]))
    model.st(alpha_dis<=t1[3])

    #（5）储能充电约束
    model.st(t1[4]*epsilon-tau*t2[4]<=-1/N_sample*alpha_ch*(zeta.sum())+(p_ch-t2[4]))
    model.st(t1[4]*epsilon-tau*t2[4]<=1/N_sample*alpha_ch*(zeta.sum())+(u_ch*p_ch_max-p_ch-t2[4]))
    model.st(alpha_ch<=t1[4])

    #（6）储能量约束
    for t in range (1,T):
        model.st(t1[5]*epsilon-tau*t2[5]<=-1/N_sample*(alpha_ch[0:t,:].sum(axis=0)+alpha_dis[0:t,:].sum(axis=0))*(zeta.sum())
                +(S_ess_max-S_ess_initial-p_ch[0:t,:].sum(axis=0)+p_dis[0:t,:].sum(axis=0)-t2[5]))
        model.st(t1[5]*epsilon-tau*t2[5]<=1/N_sample*(alpha_ch[0:t,:].sum(axis=0)+alpha_dis[0:t,:].sum(axis=0))*(zeta.sum())
                +(S_ess_max-S_ess_initial+p_ch[0:t,:].sum(axis=0)-p_dis[0:t,:].sum(axis=0)-t2[5]))

    model.st(t1>=0,t2>=0)

    # 误差调整功率平衡约束（通过样本最大值，最小值代入后的约束，文献A Wasserstein based two-stage distributionally robust optimization model 
    # for optimal operation of CCHP micro-grid under uncertainties）

    eT = np.ones(T)
    model.st((eT-alpha_dis.sum(axis = 1)+alpha_ch.sum(axis = 1)+alpha_w.sum(axis = 1)-alpha_g.sum(axis = 1))*max(zeta)<=0,
            (eT-alpha_dis.sum(axis = 1)+alpha_ch.sum(axis = 1)+alpha_w.sum(axis = 1)-alpha_g.sum(axis = 1))*min(zeta)<=0,
            -(eT-alpha_dis.sum(axis = 1)+alpha_ch.sum(axis = 1)+alpha_w.sum(axis = 1)-alpha_g.sum(axis = 1))*max(zeta)<=0,
            -(eT-alpha_dis.sum(axis = 1)+alpha_ch.sum(axis = 1)+alpha_w.sum(axis = 1)-alpha_g.sum(axis = 1))*min(zeta)<=0)


    # 调整系数约束
    model.st(alpha_g>=0,
            alpha_g<=1,
            alpha_g<=u,
            alpha_w>=0,
            alpha_w<=1,
            alpha_dis>=0,
            alpha_dis<=1,
            alpha_dis<=u_dis,
            alpha_ch>=0,
            alpha_ch<=1,
            alpha_ch<=u_ch
            )



    # 6.10 潮流约束
    start = time.time()

    # 节点导纳矩阵构建
    susceptance = 1/reactance                                       # 电纳 = 电抗倒数
    node_admittance_matrix = np.zeros((numnodes,numnodes))          # 节点导纳矩阵
    from_node = branch['From Bus'].values                           # from节点
    to_node = branch['To Bus'].values                               # to节点

    for k in range(lines):
        i = from_node[k]-1
        j = to_node[k]-1
        node_admittance_matrix[i,j] = -susceptance[k]                           # 电纳负数
        node_admittance_matrix[j,i] = node_admittance_matrix[i,j]               # 对称位置值相同
    for k in range(numnodes):
        node_admittance_matrix[k,k] = -node_admittance_matrix[k,:].sum()        # 对角线为其他元素和取负

    # 线路导纳矩阵构建
    line_admittance_matrix = np.zeros((lines,numnodes))
    for k in range(lines):
        i = from_node[k]-1
        j = to_node[k]-1
        line_admittance_matrix[k,i] = susceptance[k]
        line_admittance_matrix[k,j] = -susceptance[k]

    # 获得节点导纳矩阵的逆矩阵（在松弛节点处平衡）
    slack_bus = 8    # 松弛节点
    node_admittance_matrix = np.delete(node_admittance_matrix, slack_bus-1, axis=0)     # 去除松弛节点所在行
    node_admittance_matrix = np.delete(node_admittance_matrix, slack_bus-1, axis=1)     # 去除松弛节点所在列
    inverse_node_admittance_matrix = np.linalg.inv(node_admittance_matrix)              # 求逆矩阵
    inverse_node_admittance_matrix = np.insert(inverse_node_admittance_matrix,
                                            slack_bus-1, 0, axis=0)                  # 松弛节点所在处加行0元素
    inverse_node_admittance_matrix = np.insert(inverse_node_admittance_matrix,
                                            slack_bus-1, 0, axis=1)                  # 松弛节点所在处加列0元素
    X = inverse_node_admittance_matrix                                                  # 节点导纳矩阵的逆矩阵

    G = line_admittance_matrix@X                                                # PTDF


    generator_nodes = units['Bus #'].values         # 发电机的节点编号
    wind_nodes = np.array([3,5,7,16,21,23])         # 风电节点
    wind_nodes = wind_nodes[0:W]
    ess_nodes = np.array([4,25,50])                   # 储能节点


    sum_node_GSDF = np.zeros((T,lines))             # 每个节点的负荷注入
    load_rate_new = np.zeros(numnodes)              # 负荷节点率
    load_bus = load_bus.astype(int)                 # 负荷节点
    load_rate_new[load_bus-1] = load_rate           # 获取每个负荷节点的注入率
    G_generator = G[:,generator_nodes-1]            # 获取有负荷注入的PTDF矩阵
    G_wind = G[:,wind_nodes-1]                      # 获取风电的PTDF矩阵
    G_ess = G[:,ess_nodes-1]                        # 获取储能的PTDF矩阵


    # 获取负荷注入以及所有机组出力注入
    for t in range(T):
        sum_node_GSDF[t,:] = sum_node_GSDF[t,:] + (G*load[t]*load_rate_new).sum(axis = 1)
        model.st(sum_power_GSDF[t,:] == ((G_generator[:,:]*g[t,:]).T).sum(axis = 0) + 
                ((G_wind[:,:]*(p_w_f-p_w_cut)[t,:]).T).sum(axis = 0)+((G_ess[:,:]*(p_dis[t,:]-p_ch[t,:])).T).sum(axis = 0))


    # 潮流最终约束    
    for t in range(T):
        model.st(-capacity<=sum_power_GSDF[t,:]-sum_node_GSDF[t,:])
        model.st(sum_power_GSDF[t,:]-sum_node_GSDF[t,:]<=capacity)


    ## 7 模型求解
    solution_times = np.zeros([3,1])
    ops = np.zeros([3,1])
    # model.solve(grb)                                    # Solve the model
    # solution_times[0] = model.solution.time
    # ops[0] = model.get()
    # model.solve(cpt)
    # solution_times[1] = model.solution.time
    # ops[1] = model.get()
    # model.solve()
    # solution_times[2] = model.solution.time
    # ops[2] = model.get()
    model.solve(cpt)
    end = time.time()
    run_time = end-start

    print("代码运行时间为：", run_time, "秒")


    print("model.get\n")
    print(model.get())
    # print(p_w_f-p_w_cut.get()) 
    print("二次规划部分值:")
    print(g_alpha_g.get()@Q@g_alpha_g.get())
    print("机组启停代价:")
    print((start_cost*v.get()+down_cost*w.get()).sum())
    print("lambda*epsilon:")
    print(lambda_.get()*epsilon)
    print("一次项部分:")
    print(((b*alpha_g.get()).sum()+(alpha_w.get()*C_cut).sum())*zeta.sum())
    print("一次项中和alpha_g相关部分:")
    print((b*alpha_g.get()).sum())
    print("一次项中和alpha_w相关部分:")
    print((alpha_w.get()*C_cut).sum())
    print("燃料代价一次函数部分:")
    print((b*g.get() + c*u.get()).sum())
    print("爬坡代价:")
    print((C_up_sr*up_sr.get()+C_down_sr*down_sr.get()).sum())
    print("弃风代价:")
    print((C_cut*p_w_cut.get()).sum())
    ## 8 图像绘制
    import matplotlib.pyplot as plt

    # 示例数据
    flag = 0
    if flag:
        hours = T  # 24小时
        times = range(hours)  # 时间点 1-24 小时
        thermal_power = g.get().sum(axis=1)
        wind_power = (p_w_f-p_w_cut.get()).sum(axis=1)
        storage_discharge = p_dis.get().sum(axis=1)
        storage_charge = p_ch.get().sum(axis=1)
        cutting_wind = p_w_cut.get().sum(axis=1)
        load_curve = load
        ESS_stored = S_ess.get().sum(axis=1)[1:]


        plt.bar(times, thermal_power, color=(170/255,195/255,239/255), label='Thermal power units')
        plt.bar(times, wind_power, bottom=thermal_power, color='#ff847b', label='Wind turbines')
        plt.bar(times, storage_discharge, bottom=thermal_power + wind_power, color='#9467bd', label='Energy storage discharging')
        plt.bar(times, -storage_charge, color='#17becf', label='Energy storage charging')
        plt.bar(times, cutting_wind, bottom=thermal_power + wind_power + storage_discharge, color='#a2d2bf', label='Cutting wind')
        plt.bar(times, ESS_stored, bottom=thermal_power + wind_power + storage_discharge + cutting_wind, color=(233/255,194/255,230/255), label='ESS stored')

        # 绘制负载曲线
        plt.plot(times, load_curve, 'r--', label='Load curve')

        # 设置图例、标签和网格
        plt.legend(loc='upper left')
        plt.xlabel('Time/h')
        plt.ylabel('Power/MW')
        plt.grid(True)

        # 显示图表
        plt.show()
    return model.get(),solution_times,ops

if __name__ == '__main__':
    ## 2.输入风电误差数据，构建uncertainty set，并获取Wasserstein ball半径epsilon
    data_all = pd.read_excel('data\error_elia.xlsx')
    wind_f = data_all['WindForecast'].values
    wind_t = data_all['windTrue'].values
    wind_capacity = data_all['WindCapacity'].values
    winddata = (wind_f-wind_t)/wind_capacity*1200
    # random.seed(0)
    all_list = range(winddata.shape[0])

    random_times = 1
    nums = np.array([1000,2000,5000,8000,10000,20000,50000])
    # nums = np.array([1000,2000,3000])
    op_dro = np.zeros([random_times,len(nums)])
    op_ro = np.zeros([random_times,len(nums)])
    op_so = np.zeros([random_times,len(nums)])
    op_wdro = np.zeros([random_times,len(nums)])

    # solve_type = 3
    # op_dro = np.zeros([solve_type,random_times,len(nums)])
    # op_ro = np.zeros([solve_type,random_times,len(nums)])
    # op_so = np.zeros([solve_type,random_times,len(nums)])
    # solution_time_dro = np.zeros([solve_type,random_times,len(nums)])
    # solution_time_ro = np.zeros([solve_type,random_times,len(nums)])
    # solution_time_so = np.zeros([solve_type,random_times,len(nums)])

    for num in nums:
        for epoch in range(random_times):
            # 控制样本输入个数
            rand_index = random.sample(all_list, num)
            winddata_ex = winddata[rand_index]
            epsilon = AmbiguitySet.get_epsilon_para(winddata_ex,0,10000,0.95)
            zeta = AmbiguitySet.get_uncertainty_set(winddata_ex, 0.95, epsilon)
            print("真实数据集epsilon is",epsilon)

            model = ro.Model('DRO')         
            op_ob,solution_time,ops = WDRCC_118_Bus(model,zeta,epsilon)
            op_dro[epoch,num==nums] = op_ob
            # op_dro[:,epoch,num==nums] = ops
            # solution_time_dro[:,epoch,num==nums] = solution_time


            model = ro.Model('WDRO')         
            op_ob,solution_time,ops = WDRO_118(model,zeta,epsilon)
            op_wdro[epoch,num==nums] = op_ob


            epsilon = 0
            zeta = AmbiguitySet.get_uncertainty_set(winddata_ex, 0, epsilon)
            model = ro.Model('SO')         
            op_ob,solution_time,ops = WDRCC_118_Bus(model,zeta,epsilon)
            op_so[epoch,num==nums] = op_ob
            # op_so[:,epoch,num==nums] = ops
            # solution_time_so[:,epoch,num==nums] = solution_time

            epsilon = 80
            zeta = AmbiguitySet.get_uncertainty_set(winddata_ex, 0.95, epsilon)
            model = ro.Model('RO')
            op_ob,solution_time,ops = WDRCC_118_Bus(model,zeta,epsilon)
            op_ro[epoch,num==nums] = op_ob
            # op_ro[:,epoch,num==nums] = ops
            # solution_time_ro[:,epoch,num==nums] = solution_time

    # os.makedirs('exp_data\IEEE_118_bus_SO_RO_WDRO_WDRCC', exist_ok=True)
    # np.save('exp_data\IEEE_118_bus_SO_RO_WDRO_WDRCC\WDRCC.npy',op_dro)
    # np.save('exp_data\IEEE_118_bus_SO_RO_WDRO_WDRCC\WDRO.npy',op_wdro)
    # np.save('exp_data\IEEE_118_bus_SO_RO_WDRO_WDRCC\RO.npy',op_ro)
    # np.save('exp_data\IEEE_118_bus_SO_RO_WDRO_WDRCC\SO.npy',op_so)

    bar_width = 0.15
    x = np.arange(len(nums))

    # 设置背景颜色
    plt.figure()

    # 绘制条形图
    plt.bar(x, op_ro.mean(axis=0), width=bar_width, color='#aac3ef', label='RO', align='center')
    plt.bar(x + bar_width, op_wdro.mean(axis=0), width=bar_width, color=(233/255,194/255,230/255), label='WDRO', align='center')
    plt.bar(x + 2*bar_width, op_dro.mean(axis=0), width=bar_width, color='#ff847b', label='WDRCC', align='center')
    plt.bar(x + 3 * bar_width, op_so.mean(axis=0), width=bar_width, color='#a2d2bf', label='SO', align='center')

    # # 添加数据标签
    # for i in range(len(nums)):
    #     plt.text(i, op_ro.mean(axis=0)[i] + 2000, f"{int(op_ro.mean(axis=0)[i])}", ha='center', color='black')
    #     plt.text(i + bar_width, op_wdro.mean(axis=0)[i] + 2000, f"{int(op_wdro.mean(axis=0)[i])}", ha='center', color='black')
    #     plt.text(i + 2*bar_width, op_dro.mean(axis=0)[i] + 2000, f"{int(op_dro.mean(axis=0)[i])}", ha='center', color='black')
    #     plt.text(i + 3 * bar_width, op_so.mean(axis=0)[i] + 2000, f"{int(op_so.mean(axis=0)[i])}", ha='center', color='black')

    # 设置x轴标签和图例
    plt.xticks(x + bar_width*3/2, nums)
    plt.legend(loc='upper left')
    plt.xlabel('Sample Size', fontsize=12)
    plt.ylabel('Cost/$', fontsize=12)
    plt.ylim(8e5, 14e5)

    # 添加网格
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 显示图形
    plt.tight_layout()
    plt.show()