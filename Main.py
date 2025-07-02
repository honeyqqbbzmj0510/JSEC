import numpy as np
from time import time
from scipy.optimize import minimize
import math
from sklearn.cluster import KMeans
import yaml
import sys
import warnings
from itertools import chain, repeat
import itertools
from Mixture import create_F_W, Stable_graph, kmeans_labels
from SpectralEmbedding import compute_laplacian, optimize_view_embeddings, optimize_combined_embedding
from Run import Logger
from Data import Caltech101_7, sources, BBCSport, Orl, MSRC
from evaluation import clustering_metrics
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import csr_matrix
from scipy import stats
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
warnings.filterwarnings("ignore")

'''
=========================================================================
 |---| 描述: 算法
 |---| 参数 {*} S_list: 初始图数据
 |---| 参数 {*} gnd: 真实标签
 |---| 参数 {*} dataname: 数据集的名称
 |---| 参数 {*} order_W: 图的阶数
 |---| 参数 {*} parameters: 算法的参数，包括 \alpha、\beta、\lambda_reg
 |---| 参数 {*} type: 数据的类型
 |---| 返回 {*}
    acc, nmi, ari, f1, pur, time: 准确率、归一化互信息、调整后的兰德指数、F1 分数、纯度、时间
===========================================================================
'''

# 迭代图
def graph_obj(iter_values):
    iterations = 50  # 假设迭代次数
    if len(iter_values) < iterations:
        iter_values = list(chain(iter_values, repeat(min(iter_values), iterations - len(iter_values))))
    # 绘制迭代曲线
    plt.figure(figsize=(5, 3))
    plt.plot(range(1, iterations + 1), iter_values, marker='o', linestyle='-', color='red',
             label='Iteration Values')  # 修改颜色为红色
    plt.xlabel('Iteration Number')
    plt.ylabel('Iteration Value')
    plt.legend()
    plt.xticks(range(5, iterations + 1, 5))  # 设置横轴刻度为5的倍数
    plt.grid(False)  # 去掉参考线
    # plt.show()

# 更新矩阵
def update_A_v(nada_S, E_v, S_list, A_v, W):
    """
    更新一致特征矩阵

    参数:
    nada_S - 视图权重列表
    E_v - 不一致特征矩阵列表
    S_list - 初始图数据列表
    A_v - 一致特征矩阵列表（初始化为总相似性矩阵）
    W - 权衡参数矩阵

    返回:
    A_v - 更新后的一致特征矩阵列表
    """
    num_view = len(S_list)
    # 计算矩阵M
    M = np.zeros_like(A_v[0])
    for v in range(num_view):
        for u in range(num_view):
            M += W[v, u] * nada_S[v] * nada_S[u] * (E_v[v] @ E_v[u].T)
        M += np.diag(nada_S[v] * np.diag(E_v[v]))

    # 计算M的伪逆
    M_pinv = np.linalg.pinv(M)

    # 计算P向量
    P = np.zeros_like(A_v[0])
    for v in range(num_view):
        for u in range(num_view):
            P += W[v, u] * nada_S[v] * nada_S[u] * (S_list[v] @ A_v[u].T)
        P += S_list[v] * nada_S[v]

    # 计算vec(C^v)
    C_vec = M_pinv @ P.ravel()

    # 更新一致特征矩阵A_v
    for v in range(num_view):
        C = C_vec[v * A_v[0].size:(v + 1) * A_v[0].size].reshape(A_v[v].shape)
        C = np.maximum(C, 0)
        C = np.minimum(C, A_v[v])
        A_v[v] = csr_matrix(C)

    return A_v

def update_F_list(nada_S, lambda_reg, L_v, F_star):
    V = len(L_v)  # 视图的数量
    F_list = []
    for v in range(V):
        n = L_v[v].shape[0]  # 当前视图的样本数量
        I = np.eye(n)  # 构建单位矩阵
        # 计算更新矩阵
        A = nada_S[v] * I + lambda_reg * I + L_v[v]
        # 计算 F_v
        F_v = np.linalg.inv(A) @ (nada_S[v] * F_star[v])
        # 存储结果
        F_list.append(F_v)
    return F_list

def update_nada_S(A_v, F_list):
    V = len(A_v)  # 簇的数量
    nada_S = np.zeros(V)  # 初始化 eta_v 的数组
    U = sum(F_list[v][:] for v in range(V))
    # 对每个簇 v，计算 ||C^v - U||_F 并更新 eta_v
    for v in range(V):
        # 计算 Frobenius 范数
        frobenius_norm = np.linalg.norm(A_v[v] - U, 'fro')
        # 根据公式更新 eta_v
        nada_S[v] = 1 / (2 * np.sqrt(frobenius_norm) + 1e-5)
    return nada_S

def update_joint_embedding(F_list, nada_S, S_list,L_v):
    """
    更新联合谱嵌入矩阵 F_star

    参数：
    - F_list (list)：视图嵌入矩阵列表
    - nada_S (list)：视图权重
    - S_list (list of np.ndarray)：初始图数据列表

    返回：
    - F_star (np.ndarray)：更新后的联合谱嵌入矩阵
    """
    # 计算矩阵 N
    N = 2 * sum(nada_S[v] * F_list[v] for v in range(len(F_list)))
    for v in range(len(S_list)):
        # 更新矩阵 N
        # N += (nada_S[v] * F_list[v]) @ np.linalg.inv(L_v[v]) @ F_list[v].T
        N += np.linalg.inv(L_v[v]) @ F_list[v]
    # 对矩阵 N 进行奇异值分解
    U, _, Vt = np.linalg.svd(N, full_matrices=False)
    # 计算 F_star
    F_star = U @ (Vt+np.random.uniform(0.01))
    return F_star

def calculate_confidence_interval(data, confidence=0.95):
    """
    计算给定数据的95%置信区间
    :param data: 数据列表或numpy数组
    :param confidence: 置信度，默认为0.95
    :return: 置信区间 (lower, upper)
    """
    mean = np.mean(data)
    sem = stats.sem(data)  # 标准误差
    interval = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
    return interval

# 算法函数
def JSEC(S_list, gnd,idx, dataname, order_W=3, parameters=None, idd=0):
    """
    执行算法的主要函数

    参数：
    - S_list (list of np.ndarray)：初始图数据列表
    - gnd (np.ndarray)：真实标签
    - dataname (str)：数据集的名称
    - order_W (int)：图的阶数
    - parameters (dict)：算法的参数，包括 \alpha、\beta、\lambda_reg
    - idd (int)：参数索引

    返回：
    - acc (float)：准确率
    - nmi (float)：归一化互信息
    - ari (float)：调整后的兰德指数
    - f1 (float)：F1 分数
    - pur (float)：纯度
    - time (float)：算法运行时间
    """
    # 从参数字典中提取 alpha, beta, lambda_reg 的值
    with open('Reproduce.yaml') as f:
        cifg = yaml.load(f, Loader=yaml.FullLoader)
    parameters = cifg["parameters"]

    alpha_list = parameters["alpha"]
    beta_list = parameters['beta']
    lambda_reg_list = parameters['lambda_reg']
    alpha = float(alpha_list[idd])
    beta = float(beta_list[idd])
    lambda_reg = float(lambda_reg_list[idd])

    # 生成所有参数组合
    param_combinations = itertools.product(alpha_list, beta_list, lambda_reg_list)

    # 获取数据的维度
    N = S_list[0].shape[0]
    I = np.eye(N)
    print("Data is {}. It contains {} nodes.".format(dataname, N))

    # 获取视图的数量
    num_view = len(S_list)
    num_labels = len(np.unique(gnd))

    # 创建图的拉普拉斯矩阵或使用稳定图
    if order_W <= 10:
        FW,total_sm  = create_F_W(S_list, order=2)
    else:
        FW = Stable_graph(S_list)

    weight = [1/num_view]*num_view
    weight = np.array(weight)[:, np.newaxis] * np.ones((1, num_view))
    print("=============Initial!=============")

    # 初始化视图权重
    nada_S = [1 / num_view] * num_view
    print("Datasets====>{} Views====>{}".format(dataname, num_view))
    loss_last = 0

    # 生成一个 V*V 的矩阵 W
    W = np.full((num_view, num_view), beta)
    np.fill_diagonal(W, alpha)

    [total_similarity_matrix,total_sm] = create_F_W(S_list, order=2)
    # total_similarity_matrix = create_F_W(S_list, order=2)
    total_similarity_matrix = np.array(total_similarity_matrix)
    A_v = total_similarity_matrix
    L_v = compute_laplacian(total_similarity_matrix)
    S_origin = S_list
    S_list = total_sm
    E_v = S_origin

    # F_list = optimize_view_embeddings(L_v, lambda_reg, num_features=None)
    F_list = optimize_view_embeddings(L_v, lambda_reg, num_features=num_view)
    F_star = optimize_combined_embedding(F_list, nada_S)

    begin_time = time()

    loss_last = 0  # 初始化上一次损失为None
    obj_list = []
    iter = 0  # 初始化迭代次数为0
    maxiter = 100
    while True :
        iter += 1
        # 记录上一次的S，用于判断是否收敛
        last_S = S_list.copy()
        Loss = 0

        # 遍历每个视图，计算总损失
        for v in range(num_view):
            # 1. 多样性约束部分
            diversity_loss = 0
            for u in range(num_view):
                term = np.trace((S_list[v] - A_v[v]) @ (S_list[u] - A_v[u]).T)  # 计算 tr((S_v - A_v)(S_u - A_u)^T)
                diversity_loss += W[v, u] * nada_S[v] * nada_S[u] * term

            # 2. 高阶信息挖掘部分
            high_order_loss = 0
            diff = A_v[v] - FW[v]
            high_order_loss += nada_S[v] * (np.linalg.norm(diff, 'fro') ** 2)

            # 3. 谱聚类部分
            spectral_clustering_loss = 0
            spectral_clustering_loss += np.trace(F_list[v].T @ L_v[v] @ F_list[v])
            spectral_clustering_loss += nada_S[v] * (np.linalg.norm(F_list[v] - F_star, 'fro') ** 2)
            regularization_loss = lambda_reg * np.linalg.norm(F_list[v], 'fro') ** 2

            # 当前视图的总损失
            view_loss = diversity_loss + high_order_loss + spectral_clustering_loss + regularization_loss
            Loss += view_loss

        # 根据损失值计算容忍度（Tol），用于判断收敛
        Lossss = []
        Lossss.append(Loss)
        oder = math.log10(Loss)
        oder = int(oder)
        oder = min(5, oder)
        Tol = 1 * math.pow(10, -oder)

        if loss_last is not None and math.fabs(Loss - loss_last) <= math.fabs(Tol * Loss):
            break
        else:
            obj_list.append(Loss)

        def update_A_v(num_view, FW, nada_S, S_ij, W, alpha, beta):
            """
            更新 A_v 矩阵

            参数：
            - num_view (int)：视图数量
            - FW (np.ndarray)：混合相似图
            - nada_S (list)：视图权重
            - S_ij (list of np.ndarray)：初始图数据
            - W (np.ndarray)：权重矩阵
            - alpha (float)：参数 \alpha
            - beta (float)：参数 \beta

            返回：
            - A_v (np.ndarray)：更新后的 A_v 矩阵
            """
            # 获取初始图数据
            S =[]
            for v in range(num_view):
                S.append(np.array(S_ij[v]))

            # 计算矩阵 P 和向量 q
            P = np.zeros((num_view, num_view))
            q = np.zeros(num_view)
            # 计算 P 矩阵
            for v in range(num_view):
                for u in range(num_view):
                    P[v, u] = 2 * nada_S[v] + nada_S[v] * nada_S[u] * W[v, u]
            # 计算 q 向量
            S_F = []
            for v in range(num_view):
                S_F.append(np.array(S_ij[v]))


            q = -2 *sum(nada_S[v] * (FW[v] + (nada_S[v] * sum(W[v][u] * S_F[v] for u in range(num_view)))) for v in range(num_view))

            # 定义目标函数
            def objective(x):
                return 0.5 * x.T @ P @ x + q.T @ x
            # 设置初始值和约束
            x0 = np.zeros(num_view)
            bounds = [(0, S[v].max()) for v in range(num_view)]  # 上界为每个 S 矩阵的最大值
            # 执行优化
            result = minimize(objective, x0, bounds=bounds, method='SLSQP')
            # result = minimize(objective, bounds=bounds, method='SLSQP')

            if not result.success:
                raise ValueError("优化失败: " + result.message)
            x_opt = result.x
            # 计算并返回最终的 A_v 矩阵
            A_v = np.zeros((num_view, len(S[0])))
            for v in range(num_view):
                A_v[v] = x_opt[v] * S[v]
            return A_v

        def update_view_weights(A_v, F_list):
            V = len(A_v)  # 簇的数量
            nada_S = np.zeros(V)  # 初始化 eta_v 的数组
            U = sum(F_list[v][:] for v in range(V))
            # 对每个簇 v，计算 ||C^v - U||_F 并更新 eta_v
            for v in range(V):
                # 计算 Frobenius 范数
                frobenius_norm = np.linalg.norm(A_v[v] - U, 'fro')
                # 根据公式更新 eta_v
                nada_S[v] = 1 / (2 * np.sqrt(frobenius_norm)+1e-5)
            return nada_S

        S_ij = S_origin

        nada_S = update_view_weights(FW, A_v)
        F_list = optimize_view_embeddings(L_v, lambda_reg, num_features=num_view)
        F_star = update_joint_embedding(F_list, nada_S, S_list,L_v)

        print(f"当前迭代次数为{iter},损失值为{Loss}")
        # 如果损失变化小于容忍度，则终止迭代
        if math.fabs(Loss - loss_last) <= math.fabs(Tol * Loss) or iter==maxiter:
            break
        else:
            loss_last=Loss
            obj_list.append(Loss)

    graph_obj(obj_list) #绘制迭代曲线图
    # 进行 kmeans 聚类
    kmeans = KMeans(n_clusters=num_labels)

    kmeans.fit(F_star)
    labels = kmeans.labels_
    labels = kmeans_labels(gnd,labels,idx)

    # 记录当前时间，用于计算算法运行时间
    end_time = time()
    # 计算算法运行时间的绝对值
    Time = math.fabs(end_time - begin_time)

    # 使用指标评估聚类结果
    re = clustering_metrics(labels, gnd)
    result = re.evaluationClusterModelFromLabel()
    ac, nm, ari, f1, pur = result[0], result[1], result[2], result[3], result[4]

    # 单次返回，不计算置信区间（留给外部主函数去做）
    return ac, nm, ari, f1, pur, Time, {}


def calculate_confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=sem)
    return interval

if __name__ == '__main__':
    # 重定向标准输出到 Logger 类的实例
    sys.stdout = Logger()
    sys.stdout.show_version()

    # 加载 YAML 配置
    with open('Reproduce.yaml') as f:
        cifg = yaml.load(f, Loader=yaml.FullLoader)

    parameters = cifg["parameters"]
    datasets = cifg["datasets"]

    # 遍历数据集
    for idx, data in enumerate(datasets):
        Switcher = {
            0: sources,
            1: BBCSport,
            2: MSRC,
            3: Orl,
            4: Caltech101_7
        }
        S_list, gnd = Switcher[idx]()

        # 多次运行
        acc_list, nmi_list, ari_list, f1_list, pur_list, time_list = [], [], [], [], [], []
        num_runs = 1  # 运行次数

        for run_id in range(num_runs):
            ac, nm, ari, f1, pur, Time, _ = JSEC(S_list, gnd, idx, dataname=data, order_W=3, parameters=None, idd=0)  # 这里把 idd 固定为 0
            acc_list.append(ac)
            nmi_list.append(nm)
            ari_list.append(ari)
            f1_list.append(f1)
            pur_list.append(pur)
            time_list.append(Time)

        # 计算95%置信区间
        acc_ci = calculate_confidence_interval(acc_list)
        nmi_ci = calculate_confidence_interval(nmi_list)
        ari_ci = calculate_confidence_interval(ari_list)
        f1_ci = calculate_confidence_interval(f1_list)
        pur_ci = calculate_confidence_interval(pur_list)
        time_ci = calculate_confidence_interval(time_list)

        # 打印最终结果
        print("Datasets: {} \n"
              "ACC: {:.4f} (CI: {:.4f} ~ {:.4f})\n"
              "NMI: {:.4f} (CI: {:.4f} ~ {:.4f})\n"
              "PUR: {:.4f} (CI: {:.4f} ~ {:.4f})\n"
              "ARI: {:.4f} (CI: {:.4f} ~ {:.4f})\n"
              "F1: {:.4f} (CI: {:.4f} ~ {:.4f})\n"
              "Time: {:.4f} (CI: {:.4f} ~ {:.4f})\n".format(
            data,
            np.mean(acc_list), acc_ci[0], acc_ci[1],
            np.mean(nmi_list), nmi_ci[0], nmi_ci[1],
            np.mean(pur_list), pur_ci[0], pur_ci[1],
            np.mean(ari_list), ari_ci[0], ari_ci[1],
            np.mean(f1_list), f1_ci[0], f1_ci[1],
            np.mean(time_list), time_ci[0], time_ci[1]
        ))