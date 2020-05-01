import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy.special import expit

from logistic_regression import LogisticRegression

breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.scatter(X[:, 0], y, color='b', marker='o', s=30)
plt.show()

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


logistic_regression_model = LogisticRegression(learning_rate=0.0001, n_iterations=1000)
logistic_regression_model.fit(X_train, y_train)
predictions = logistic_regression_model.predict(X_test)

print("Logistic Regression - Classification Accuracy: ", accuracy(y_test, predictions))
