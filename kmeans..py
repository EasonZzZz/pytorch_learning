import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 生成数据
X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)
# 绘制数据
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()