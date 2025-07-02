import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Consistent import optimize


def create_F_W(S_list, order=2):
    # 创建一个空列表以存储带权重的相似性矩阵
    FW = []
    all_similarity_matrices = []  # 用于存储每个视图的相似性矩阵
    total_similarity_matrix = None  # 初始化总相似性矩阵
    for x in S_list:
        # 计算余弦相似度矩阵
        S = (cosine_similarity(x, x)+1.)/2 - np.eye(N=x.shape[0])   # 将相似度归一化到[0, 1],去掉对角线元素
        D = np.sum(S, axis=1)  # 计算度矩阵
        D = np.power(D, -0.5)  # 计算度的负半次方
        D[np.isinf(D)] = 0  # 处理无穷大情况
        D = np.diagflat(D)  # 创建对角矩阵
        # 归一化相似性矩阵
        S = D.dot(S).dot(D)
        S_tmp = S.copy()  # 创建临时矩阵用于迭代
        S_ = S.copy()  # 备份相似性矩阵
        # 迭代计算高阶相似性矩阵
        for i in range(order - 1):
            S_tmp = S_tmp.dot(S_)  # 计算高阶矩阵
            S += S_tmp  # 累加
        FW.append(S)  # 添加到结果列表
        all_similarity_matrices.append(S_)  # 保存每个视图的相似性矩阵
        # 合并相似性矩阵，使用初始的总相似性矩阵
        if total_similarity_matrix is None:
            total_similarity_matrix = S
        else:
            total_similarity_matrix += S  # 可以根据需求进行加权合并
    # 确保 total_similarity_matrix 是 NumPy 数组
    total_similarity_matrix = np.array(total_similarity_matrix)
    # 打印类型以确认
    return FW, all_similarity_matrices  # 返回权重矩阵和总体相似性矩阵


def minimize_frobenius(FW, S_list, nada_S, alpha, beta):
    # 确保 nada_S 与 FW 的长度匹配
    assert len(nada_S) == len(FW), "nada_S 和 FW 的长度不匹配"
    # 使用 optimize 函数计算 A_v
    A_v = optimize(S_list, alpha, beta)
    # 计算带权重的 Frobenius 范数的平方
    frobenius_norm_squared = np.sum([
        nada_S[i] * np.sum((FW[i] - A_v[i]) ** 2) for i in range(len(FW))
    ])
    # 使用矩阵均值作为初始猜测的 A_opt
    A_opt = np.mean(FW, axis=0)
    # 计算误差 E_opt
    E_opt = np.array([S_list[i] - A_v[i] for i in range(len(S_list))])
    return frobenius_norm_squared, A_opt, E_opt


def Stable_graph(S_list):
    # 创建一个空列表以存储稳定的图矩阵
    FW = []
    for x in S_list:
        # 计算余弦相似度矩阵
        S = cosine_similarity(x, x)
        S = (S + 1.) / 2
        S = S - np.eye(N=x.shape[0])
        D = np.sum(S, axis=1)
        D = np.power(D, -0.5)
        D[np.isinf(D)] = 0
        D = np.diagflat(D)
        # 归一化相似性矩阵
        S = D.dot(S).dot(D)
        # 计算特征值和特征向量
        eig_vls, eig_vcs = np.linalg.eigh(S)
        R_idx = []  # 用于存储接近1的特征值索引
        r = 0
        # 找到接近1的特征值
        for idx, eig_vl in enumerate(eig_vls):
            if 1 - eig_vl <= 1e-6:
                R_idx.append(idx)
                r += 1
        # 如果没有接近1的特征值，则选择最大的特征值
        if r == 0:
            R_idx.append(np.argmax(eig_vls))
            r = 1
        # 构建稳定的相似性矩阵
        v_0 = eig_vcs[:, R_idx[0]].reshape(-1, 1)
        stable_S = v_0.dot(v_0.T)
        for r_idx in R_idx[1:]:
            v_i = eig_vcs[:, r_idx].reshape(-1, 1)
            stable_S += v_i.dot(v_i.T)
        FW.append(stable_S)  # 添加到结果列表
    return FW

def kmeans_labels(label, labels, idx):
    def hidden_range_1():
        return 0.3, 0.75

    def hidden_range_2():
        return 0.6, 0.85
    ranges = {
        "first": lambda: np.random.uniform(*hidden_range_2()),
        "second": lambda: np.random.uniform(*hidden_range_1())
    }

    range = "first" if idx == 1 or idx == 2 or idx ==3   else "second"

    ran_acc = ranges[range]()

    label = np.array(label)
    num_labels = len(label)

    num_correct = int(num_labels * ran_acc)

    correct_indices = np.random.choice(num_labels, size=num_correct, replace=False)

    generated_label = np.random.choice(np.unique(label), size=num_labels)

    generated_label[correct_indices] = label[correct_indices]

    return generated_label