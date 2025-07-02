import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# K-means++聚类
# 读取 Excel 文件
data_path = r"C:\Users\asus\Desktop\产品创新.xlsx"
df = pd.read_excel(data_path)
# 处理缺失值
df = df.dropna(subset=["FAC1", "FAC2", "FAC3", "FAC4", "FAC5", "FAC6", "FAC7"])
# 选择需要聚类的列
columns = ["FAC1", "FAC2", "FAC3", "FAC4", "FAC5", "FAC6", "FAC7"]
X = df[columns]
# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 手肘法确定最佳聚类数
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.show()
# 基于手肘法结果选择k（假设最佳k=5）
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
# 输出聚类结果统计
print("各簇样本量：\n", df["Cluster"].value_counts())
cluster_stats = df.groupby("Cluster")[columns].mean()
print("\n各簇因子均值：\n", cluster_stats)
# 使用 t-SNE 进行降维可视化
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
# 绘制 t-SNE 结果
plt.figure(figsize=(8, 6))
for cluster in range(k):
    plt.scatter(X_tsne[df["Cluster"] == cluster, 0],
                X_tsne[df["Cluster"] == cluster, 1],
                label=f'Cluster {cluster}')
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("K-Means Clustering with t-SNE Visualization")
plt.legend()
plt.show()


# 因子回归影响系数
from sklearn.linear_model import LogisticRegression
X_factors = df[[ "FAC1","FAC2", "FAC3", "FAC4", "FAC5", "FAC6", "FAC7"]]
y_cluster = df["Cluster"]
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
log_reg.fit(X_factors, y_cluster)
print("因子影响程度（回归系数）:")
print(pd.DataFrame(log_reg.coef_, columns=columns, index=[f"Cluster {i}" for i in range(5)]))



# 决策树
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_factors, y_cluster)
plt.figure(figsize=(12, 6))
plot_tree(tree, feature_names=columns, class_names=[str(i) for i in range(5)], filled=True)
plt.show()


# 卡方检验+K-W检验
import pandas as pd
import scipy.stats as stats
# 1️⃣ 读取 Excel 文件
file_path = r'C:\Users\asus\Desktop\产品创新.xlsx'  # 确保文件路径正确
df = pd.read_excel(file_path)
# 确保类别列和用户特征列存在
category_col = '类别'  # 类别列名
features = ['年龄_分类', '性别', '职业类型', '所在地区', '是否常喝']  # 需要分析的分类变量
# 2️⃣ 进行卡方检验（适用于分类变量）
print("\n📊 卡方检验结果（Chi-square Test）:")
for feature in features:
    cross_tab = pd.crosstab(df[feature], df[category_col])  # 交叉表
    chi2, p, dof, expected = stats.chi2_contingency(cross_tab)  # 计算卡方检验
    print(f"{feature}: Chi-square={chi2:.2f}, p-value={p:.4f}")
# 3️⃣ 进行 Kruskal-Wallis H 检验（适用于分类变量）
print("\n📈 Kruskal-Wallis H 检验结果:")
for feature in features:
    if feature in df.columns:
        groups = [df[df[category_col] == cat][feature].dropna() for cat in df[category_col].unique()]

        if all(len(g) > 1 for g in groups):  # 确保每组至少有两个数据点
            h_stat, p_val = stats.kruskal(*groups)
            print(f"{feature}: H-statistic={h_stat:.2f}, p-value={p_val:.4f}")
        else:
            print(f"{feature}: 数据不足，无法进行 Kruskal-Wallis H 检验")




# 年龄段
from matplotlib import rcParams
from matplotlib import font_manager
# 设置字体路径，确保中文显示
font_path = r'C:\Users\asus\Desktop\市调2025\代码\STKAITI.ttf'  # 或者你自己的 ttf 字体文件路径
prop = font_manager.FontProperties(fname=font_path)
rcParams['font.sans-serif'] = [prop.get_name()]  # 使用加载的字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 读取 Excel 文件的 sheet5
file_path = r'C:\Users\asus\Desktop\产品创新.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet5')
# 提取前5行数据
columns = ['健康传统导向型','低度花香偏好型','创新技术追求型','品质风味驱动型','保守稳健型']
data = np.array([df[col].iloc[:5].values for col in columns])  # 转换为 NumPy 数组 (5,5)
labels = ['18-24岁', '25-34岁', '35-50岁', '51-64岁', '>65岁']
# 设置柱形图参数
x = np.arange(len(columns))  # X轴位置
width = 0.15  # 柱子的宽度
colors = ["#ADD8E6", "#87CEFA", "#4682B4", "#1E3A5F", "#0F1D33"]  # 五个年龄段颜色
# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))
# 绘制分组柱形图
for i in range(len(labels)):
    ax.bar(x + i * width - (2 * width), data[:, i], width, label=labels[i], color=colors[i])
# 设置X轴
ax.set_xticks(x)
ax.set_xticklabels(columns, fontsize=14)  # X轴标签
ax.tick_params(axis='y', labelsize=14)  # 纵轴刻度字体大小
ax.set_title("不同人群在各年龄段的分布情况", fontsize=16)  # 标题
ax.legend(fontsize=16)
plt.savefig("年龄段.svg", format='svg', bbox_inches='tight')
ax.legend()  # 添加图例
# 显示图形
plt.show()



# 饮酒频次
# 提取6-10行数据
data = np.array([df[col].iloc[5:10].values for col in columns])  # 取第6到10行，共5行
labels = ['每天', '每周3-5次', '每周1-2次', '每月1-3次', '从不']
# 设置柱形图参数
x = np.arange(len(columns))  # X轴位置
width = 0.15  # 柱子的宽度
colors = ["#ADD8E6", "#87CEFA", "#4682B4", "#1E3A5F", "#0F1D33"]  # 每个频次对应不同颜色
# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))
# 绘制分组柱形图
for i in range(len(labels)):
    ax.bar(x + i * width - (2 * width), data[:, i], width, label=labels[i], color=colors[i])
# 设置 X 轴
ax.set_xticks(x)
ax.set_xticklabels(columns, fontsize=14)  # X轴标签字体大小
# 设置 Y 轴
ax.tick_params(axis='y', labelsize=14)  # 纵轴刻度字体大小
# 设置标题
ax.set_title("不同人群在饮酒频次的分布情况", fontsize=16)
ax.legend(fontsize=12)
# 保存图像
plt.savefig("频次.svg", format='svg', bbox_inches='tight')
# 显示图形
plt.show()


# 职业分布
# 提取15-18行数据
data = np.array([df[col].iloc[14:18].values for col in columns])  # 取第15到18行，共4行
labels = ['学生', '工作', '其他']  # 只有4个类别
# 设置柱形图参数
x = np.arange(len(columns))  # X轴位置
width = 0.15  # 柱子的宽度
colors = ["#ADD8E6", "#4682B4", "#0F1D33"]  # 4 种颜色
# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))
# 绘制分组柱形图
for i in range(len(labels)):
    ax.bar(x + i * width - (1.5 * width), data[:, i], width, label=labels[i], color=colors[i])
# 设置 X 轴
ax.set_xticks(x)
ax.set_xticklabels(columns, fontsize=14)  # X轴标签字体大小
# 设置 Y 轴
ax.tick_params(axis='y', labelsize=14)  # 纵轴刻度字体大小
# 设置标题
ax.set_title("不同人群在职业的分布情况", fontsize=16)
ax.legend(fontsize=12)
# 保存图像
plt.savefig("职业.svg", format='svg', bbox_inches='tight')
# 显示图形
plt.show()


#汾酒集团营收
# 读取 Excel 文件中的 Sheet9
file_path = r"C:\Users\asus\Desktop\产品创新.xlsx"
data = pd.read_excel(file_path, sheet_name='Sheet9')
# 获取第一列和第二列的数据
x = data.iloc[:, 0]  # 第一列
y = data.iloc[:, 1]  # 第二列
# 绘制折线图
plt.plot(x, y, marker='o', linestyle='-', color="#2A5D8C", label='Data Line')#, "#D4AF37"
# 添加标题和标签
plt.title('汾酒2009-2023年营收数据')
plt.ylabel('营业收入(亿元)')
plt.savefig("营收.svg", format='svg', bbox_inches='tight')
# 显示图形
plt.show()