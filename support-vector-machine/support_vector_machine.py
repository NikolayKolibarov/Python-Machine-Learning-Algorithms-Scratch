import numpy as np


class SupportVectorMachine:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        transformed_y = np.where(y <= 0, -1, 1)

        # init parameters
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            for i, x_i in enumerate(X):
                # compute gradients and update parameters
                if transformed_y[i] * (np.dot(x_i, self.w) - self.b) >= 1:
                    dw = 2 * self.lambda_param * self.w
                    w_update = self.w - np.dot(self.learning_rate, dw)

                    self.w = w_update
                else:
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, transformed_y[i])
                    w_update = self.w - np.dot(self.learning_rate, dw)

                    db = transformed_y[i]
                    b_update = self.b - np.dot(self.learning_rate, db)

                    self.w = w_update
                    self.b = b_update

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
