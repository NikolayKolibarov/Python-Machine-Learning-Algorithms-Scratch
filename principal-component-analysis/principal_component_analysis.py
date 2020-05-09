import numpy as np


class PrincipalComponentAnalysis:

    def __init__(self, n_components):
        self.n_components = n_components
        self.n_components = None
        self.mean = None

    def fit(self, X):
        # mean
        self.mean = np.mean(X, axis=0)

        # covariance
        # row = 1 sample, columns = feature
        cov = np.cov((X - self.mean).T)

        # eigenvectors, eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # sort eigenvectors
        eigenvectors = eigenvectors.T
        indecies = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indecies]
        eigenvectors = eigenvectors[indecies]

        # store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        return np.dot(X - self.mean, self.components.T)
