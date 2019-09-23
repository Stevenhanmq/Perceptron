# Input: numpy matrix X of features, with n rows (samples), d columns (features)
#       X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
#       y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector theta of d rows, 1 column
import cvxopt as co
import numpy as np

def run(X,y):
    # Your code goes here
    (n, d) = np.shape(X)
    H = np.identity(d)
    f = np.zeros((d, 1))
    b = np.negative(np.ones((n, 1)))
    A = np.zeros((n, d))
    
    
    for i in range(n):
        for j in range(d):
            A[i][j] = np.negative(y[i] * X[i][j])

    co.solvers.options['show_progress'] = False
    
    theta = np.array(co.solvers.qp(co.matrix(H,tc='d'),co.matrix(f,tc='d'),
                                   co.matrix(A,tc='d'),co.matrix(b,tc='d'))['x'])
                    
    return theta
