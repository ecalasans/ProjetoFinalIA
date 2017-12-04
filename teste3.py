import numpy as np

X = np.array([
    [1,1],
    [1,0],
    [0,1],
    [0, 0],
    
])

y = np.array([1,0,0,0])

def perceptron_sgd_plot(X, Y):
    '''
    train perceptron and plot the total loss in each epoch.
    
    :param X: data samples
    :param Y: data labels
    :return: weight vector as a numpy array
    '''
    w = np.zeros(len(X[0]))
    eta = 1
    n = 30
    errors = []

    for t in range(n):
        total_error = 0
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                total_error += (np.dot(X[i], w)*Y[i])
                w = w + eta*X[i]*Y[i]
        errors.append(total_error*-1)
        
    # plt.plot(errors)
    # plt.xlabel('Epoch')
    # plt.ylabel('Total Loss')
    
    return w


print(perceptron_sgd_plot(X,y))

