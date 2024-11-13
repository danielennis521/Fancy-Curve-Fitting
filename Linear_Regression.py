class L1LinearRegression():
    def __init__(self, learning_rate=1e-6, max_iterations=1000, L1_penalty=1, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.L1_penalty = L1_penalty
        self.tol = tol
        self.a = np.array([])
        self.n = 0  # number of observations
        self.m = 0  # number of regressors

    def fit(self, x, y, a=None):
        
        self.x = np.array(x)
        self.x = np.insert(self.x, 0, 1, axis=1)
        self.x_norm = self.normalize(self.x.copy())
        self.y = np.array(y)
        self.n, self.m = self.x.shape
        if a is None:
            self.a = np.zeros(self.m)
        else:
            self.a = a

        for i in range(self.max_iterations):
            a = self.a.copy()
            self.a -= self.learning_rate * (1/self.n) * self.gradient()
            if sum(abs(a - self.a)) <= self.tol: break

        return a
        
    def gradient(self):
        penalty = self.L1_penalty * np.sign(self.a)
        penalty[0] = 0
        return np.dot(self.x_norm.T, np.dot(self.x, self.a) - self.y) + 0.5*penalty

    def normalize( self, X ) :
        print(np.mean( X[:, 1:], axis = 0 ))
        print(np.std( X[:, 1:], axis = 0 ))
        X[:, 1:] = ( X[:, 1:] - np.mean( X[:, 1:], axis = 0 ) ) / np.std( X[:, 1:], axis = 0 )
        return X
