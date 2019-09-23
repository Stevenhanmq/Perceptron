# Input: maximum number of iterations L
# 		numpy matrix X of features, with n rows (samples), d columns (features)
# 			X[i,j] is the j-th feature of the i-th sample
# 		numpy vector y of labels, with n rows (samples), 1 column
# 			y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector theta of d rows, 1 column
# 		number of iterations that were actually executed (iter+1)
import numpy as np
def run(L, X, y):
	#your code goes here
	(n,d) = np.shape(X)
	theta = np.zeros((d,1))
	for iter in range(L):
		all_points_classified_correcrly = True
		for t in range(n):
			if(y[t] * np.dot(np.array(theta)[:,0], X[t])) <= 0:
				Y = np.array(X[t])
				Y = np.reshape(Y, (d, 1))
				theta = np.add(theta, y[t] * Y)		
			all_points_classified_correcrly = False
			#endif
		#endfor
		if all_points_classified_correcrly:
			break
	#endfor
	return theta, iter+1

