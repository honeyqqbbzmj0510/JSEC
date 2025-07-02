import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

def pad_s_list(S_list):
    max_rows = max(s.shape[0] for s in S_list)
    max_cols = max(s.shape[1] for s in S_list)
    padded_S_list = np.zeros((len(S_list), max_rows, max_cols))
    for i, s in enumerate(S_list):
        if isinstance(s, csr_matrix):  # 检查是否为稀疏矩阵
            padded_S_list[i, :s.shape[0], :s.shape[1]] = s.toarray()
        else:  # 如果是 ndarray，直接赋值
            padded_S_list[i, :s.shape[0], :s.shape[1]] = s
    return padded_S_list

def hadamard_product(A, B):
    # 计算两个矩阵的哈达玛积
    return A * B


def objective(params, padded_S_list, alpha, beta):
    V, n, m = padded_S_list.shape
    # 将平展的参数还原为三维矩阵
    A_v = params.reshape(V, n, m)
    objective_value = 0
    # 计算误差矩阵
    E_v = padded_S_list - A_v
    for v in range(V):
        for u in range(V):
            if v != u:
                # 计算不同矩阵之间的哈达玛积并累加
                term = np.sum(hadamard_product(E_v[v], E_v[u]))
                objective_value += alpha * term
        # 计算自身的哈达玛积并累加
        term = np.sum(hadamard_product(E_v[v], E_v[v]))
        objective_value += beta * term
    return objective_value


def optimize(S_list, alpha, beta):
    # 将输入的稀疏矩阵列表填充为统一形状的矩阵
    padded_S_list = pad_s_list(S_list)
    V, n, m = padded_S_list.shape
    # 随机初始化参数
    initial_params = np.random.randn(V * n * m)
    # 优化目标函数
    result = minimize(
        fun=objective,
        x0=initial_params,
        args=(padded_S_list, alpha, beta),
        method='CG'
    )
    optimized_params = result.x
    # 还原为三维矩阵
    A_v = optimized_params.reshape(V, n, m)
    E_v = padded_S_list - A_v  # 计算最终的误差
    return A_v, E_v, result.fun