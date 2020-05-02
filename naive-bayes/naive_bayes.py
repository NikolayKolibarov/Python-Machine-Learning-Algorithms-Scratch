import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # init mean, var, priors
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.variance = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for c in self.classes:
            X_c = X[c == y]
            self.mean[c, :] = X_c.mean(axis=0)
            self.variance[c, :] = X_c.var(axis=0)
            # frequency of class
            self.priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self.predict_x(x) for x in X]
        return y_pred

    def predict_x(self, x):
        posteriors = []

        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            class_conditional = np.sum(np.log(self.probability_density_function(i, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def probability_density_function(self, class_i, x):
        mean = self.mean[class_i]
        variance = self.variance[class_i]
        numerator = np.exp(- (x - mean) ** 2 / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator
