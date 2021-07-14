import numpy as np
import cvxopt
from extras import create_dataset , plot_contour
def linear_kernel(x,z):
    return np.dot(x,z.T)

def polynomial(x,z,p= 5):
    return (1+np.dot(x,z.T)) ** p 
def gaussian(x,z,sigma = 0.1):
    return np.exp(-np.linalg.norm(x-z, axis = 1)**2/ 2*(sigma**2))

class SVM():
    def __init__(self, kernel = gaussian, C= 1):
        self.kernel = kernel
        self.C = C

    def fit(self,X,y):
        self.X = X 
        self.Y = y
        m , n = X.shape

        #calculating the kernel
        self.K = np.zeros((m,n))

        for i in range(m):
            self.K[i,:] = self.kernel(X[i,np.newaxis],
                                        self.X)

        #to obtain the alpha value
        P = cvxopt.matrix(np.outer(y,y)*self.K)
        q = cvxopt.matrix(-np.ones((m,1)))
        G = cvxopt.matrix(np.vstack((np.eye(m)*-1, np.eye(m))))
        h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m)*self.C)))

        A = cvxopt.matrix(y, (1,m),'d')
        b = cvxopt.matrix(np.zeros(1))

        cvxopt.solvers.options['show_progress'] = False

        sol = cvxopt.solvers.qp(P,q,G,h,A,b)

        self.alphas = np.array(sol['x'])
    def predict(self,X):
        y_predict = np.zeros((X.shape[0]))
        sv = self.get_parameters(self.alphas)

        for i in range(X.shape[0]):
            y_predict[i] = np.sum(self.alphas[sv]*self.y[sv,np.newaxis] *
            self.kernel(X[i]), self.X[sv][:,np.newaxis])

        return np.sign(y_predict + self.b)

    def get_parameters(self,alphas):
        threshold = 1e-4

        sv = ((alphas > threshold) * (alphas < self.C)).flatten()

        self.w = np.dot(self.X[sv].T, alphas[sv]*self.y[sv,np.newaxis])
        self.B = np.mean(self.y[sv,np.newaxis] - 
                        self.alphas[sv]*self.y[sv,np.newaxis]*self.K[sv,sv][:,np.newaxis])

        return sv
if __name__ == '__main__':
    np.random.seed(1)
    X,y = create_dataset(N = 50)

    svm = SVM(kernel=gaussian).fit(X,y)
    y_pred = svm.predict(X)

    print(f'acc => {sum(y == y_pred)/y.shape[0]}' )