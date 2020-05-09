from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

from principal_component_analysis import PrincipalComponentAnalysis

data = datasets.load_iris()
X, y = data.data, data.target

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

pca = PrincipalComponentAnalysis(2)
pca.fit(X)
X_projected = pca.transform(X)

print('Shape of X: ', X_train.shape)
print('Shape of transformed X: ', X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(x1, x2, c=y, edgecolor='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis', 3))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()
