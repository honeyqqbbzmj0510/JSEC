import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

def get_adjacency_matrix_from_matrix(S_list):
    valid_rows = [row for row in S_list if isinstance(row, (list, np.ndarray)) and len(row) > 0]
    if not valid_rows:
        raise ValueError("S_list must contain at least one non-empty row.")
    num_rows = len(valid_rows)
    num_cols = max(len(row) for row in valid_rows)
    adj_matrix = np.zeros((num_rows, num_cols), dtype=int)
    for i in range(num_rows):
        for j in range(len(valid_rows[i])):
            index = valid_rows[i][j]
            if (np.issubdtype(type(index), np.number) and index >= 0 and index < num_rows):
                adj_matrix[i][index] = 1
                if index < num_cols:
                    adj_matrix[index][i] = 1
    return adj_matrix


def compute_laplacian(total_similarity_matrix1):
    L_v_list = []
    for i in range(total_similarity_matrix1.shape[0]):
        total_similarity_matrix = total_similarity_matrix1[i,:]
        # 检查类型
        if isinstance(total_similarity_matrix, tuple):
            total_similarity_matrix = total_similarity_matrix[0]  # 选择合适的元素
        # 确保是 NumPy 数组
        if not isinstance(total_similarity_matrix, np.ndarray):
            raise ValueError("total_similarity_matrix should be a NumPy array.")
        # 确保是二维数组
        if total_similarity_matrix.ndim == 1:
            total_similarity_matrix = total_similarity_matrix.reshape(-1, 1)
        degree_values = np.sum(total_similarity_matrix, axis=1)  # 每一行的和
        D_v = np.diag(degree_values)
        L_v = D_v - total_similarity_matrix
        L_v_list.append(L_v)
    return L_v_list




def optimize_view_embeddings(L_v, lambda_reg, num_features):
    """
    优化每个视图的嵌入矩阵以最小化目标函数。

    参数:
    L_v : list
        拉普拉斯矩阵列表
    lambda_reg : float
        正则化参数
    num_features : int
        嵌入的特征数量

    返回:
    F_list : list
        优化后的嵌入矩阵列表
    """
    F_list = []
    for L in L_v:
        # 确保 L 是稀疏矩阵且是方形
        if not hasattr(L, 'shape') or len(L.shape) != 2 or L.shape[0] != L.shape[1]:
            raise ValueError("Each Laplacian matrix must be a square 2D matrix.")
        # 获取嵌入矩阵的形状
        F_v_shape = (L.shape[0], num_features)
        def objective_function(F_v_flat):
            # 将一维数组转换为二维矩阵
            F_v = F_v_flat.reshape(F_v_shape)
            # 计算目标函数的迹项
            trace_term = np.trace(F_v.T @ L @ F_v)
            # 计算 Frobenius 范数的平方
            frobenius_norm_squared = np.sum(F_v ** 2)
            return trace_term + lambda_reg * frobenius_norm_squared

        # 梯度函数
        def gradient(F_v_flat):
            F_v = F_v_flat.reshape(F_v_shape)
            # 计算目标函数对 F_v 的梯度
            grad = 2 * L @ F_v + 2 * lambda_reg * F_v
            return grad.flatten()

        # 求解特征值和特征向量
        try:
            eigenvalues, eigenvectors = eigsh(L, k=num_features, which='SM')
        except Exception as e:
            print(f"Error computing eigenvalues/eigenvectors: {e}")
            continue  # 跳过这一轮
        initial_F_v = eigenvectors[:, :num_features]  # 取前 num_features 个特征向量
        initial_F_v_flat = initial_F_v.flatten()  # 展平为一维数组
        # 优化
        # result = minimize(objective_function, initial_F_v_flat, method='BFGS')

        # 使用BFGS进行优化并提供梯度
        result = minimize(objective_function, initial_F_v_flat, method='L-BFGS-B', jac=gradient,
                          options={'maxiter': 10, 'gtol': 1e-3})

        if result.success:
            F_v = result.x.reshape(F_v_shape)  # 获取优化结果
            F_list.append(F_v)  # 添加到结果列表
        else:
            print(f"Optimization failed for one view: {result.message}")
            F_list.append(np.zeros(F_v_shape))  # 或者根据需要处理失败情况
    return F_list


def optimize_combined_embedding(F_list, nada_S):
    """
    优化得到联合谱嵌入矩阵 F*。
    参数:
    F_list : list
        各视图的嵌入矩阵列表
    nada_S : list
        权重列表
    返回:
    F_star : ndarray
        优化后的联合嵌入矩阵
    """
    def objective_function(F_star_flat):
        F_star = F_star_flat.reshape(F_list[0].shape)  # 将一维数组转换为矩阵
        objective_value = 0
        for F_v, nada_S_value in zip(F_list, nada_S):
            # 累加目标函数值
            objective_value += nada_S_value * np.linalg.norm(F_v - F_star, 'fro') ** 2
        return objective_value
    # F_star 的形状与 F_list 中的矩阵一致
    F_star_shape = F_list[0].shape
    initial_F_star = np.zeros(F_star_shape).flatten()  # 初始化 F_star
    # 使用 BFGS 方法优化目标函数
    result = minimize(objective_function, initial_F_star, method='BFGS')
    # 计算优化后的 F_star
    F_star = result.x.reshape(F_star_shape)  # 还原为矩阵
    return F_star