'''
@ author: 2016024793 김유진
DataScience Term Project: Predicting movie ratings
- Matrix Factorization
'''
import numpy as np
import math

class MatrixFactorization():
    def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):
        '''
        - R: rating matrix
        - k: latent parameter
        - learning_rate: alpha on weigth update
        - reg_param: beta on weight update (regulariztion parameter)
        - epochs: training epochs
        - verbose: print status
        '''
        self._R = R
        self._num_users, self._num_itmes = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose

    def fit(self):
        '''
        training Matrix Factorization: Update matrix latent weight and bias
        - returns training_process 
        '''
        # initialize latent features
        self._P = np.random.normal(size=(self._num_users, self._k))
        self._Q = np.random.normal(size=(self._num_itmes, self._k))

        # initialize biases
        self._b_P = np.zeros(self._num_users)
        self._b_Q = np.zeros(self._num_itmes)
        self._b = np.mean(self._R[np.where(self._R != 0)])

        self._training_process = []
        
        if self._verbose == True:
            print("Training start with hyperparameters:")
            print("learning rate: %.4f, reg parameter: %.4f, epochs: %d" % (self._learning_rate, self._reg_param, self._epochs))

        for epoch in range(self._epochs):
            for i in range(self._num_users):
                for j in range(self._num_itmes):
                    # train only with rated data
                    if self._R[i, j] > 0:
                        self.gradient_descent(i, j, self._R[i, j], epoch)
            
            cost = self.cost()
            self._training_process.append((epoch, cost))
            
            # print status
            if self._verbose == True and ((epoch+1) % 10 == 0):
                print("Iteration: %03d ; cost = %f" % (epoch+1, cost))

            '''
            if epoch != 0:
                cost_diff = self._training_process[-2][1] - cost
                if cost_diff < 0.0000001:
                    print("cost difference sufficiently small! iteration: {}".format(epoch + 1))
                    self._epochs = epoch + 1
                    break
            '''
                
    def cost(self):
        ''' 
        compute root mean square error
        - returns RMSE cost
        '''
        xi, yi = self._R.nonzero()
        n = len(xi)
        predicted = self.get_complete_matrix()
        cost = 0

        for x, y in zip(xi, yi):
            cost += (self._R[x,y] - predicted[x, y])**2
        
        return np.sqrt(cost) / n

    def gradient_descent(self, i, j, rating, e):
        '''
        gradient descent function
        - i: user index of matrix
        - j: item index of matrix
        - rating: rating of (i,j)
        '''
        prediction = self.get_prediction(i, j)
        error = rating - prediction
        
        # update bias
        self._b_P[i] += self._learning_rate * (error - self._reg_param * self._b_P[i])
        self._b_Q[j] += self._learning_rate * (error - self._reg_param * self._b_Q[j])

        # update latent feature
        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
        self._P[i, :] += self._learning_rate * dp
        self._Q[j, :] += self._learning_rate * dq
         
        
    def get_prediction(self, i, j):
        return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)

    def get_complete_matrix(self):
        '''
        compute complete matrix R^
        '''
        return self._b + self._b_P[:, np.newaxis] + self._b_Q[np.newaxis:, ] + self._P.dot(self._Q.T)

    def get_rounded_matrix(self):
        p = self.get_complete_matrix()
        
        for i in range(self._num_users):
            for j in range(self._num_itmes):
                r = p[i,j]
                if r < 0. or math.isnan(r) or math.isinf(r):
                    p[i,j] = 0.
        return p

    def print_results(self):
        print("learning rate: %.4f, reg parameter: %.4f, epochs: %d" % (self._learning_rate, self._reg_param, self._epochs))
        print("User Latent P:")
        print(self._P.shape)
        print("Item Latent Q:")
        print(self._Q.T.shape)
        print("P x Q:")
        print(self._P.dot(self._Q.T).shape)
        print("bias:")
        print(self._b.shape)
        print("User Latent bias:")
        print(self._b_P.shape)
        print("Item Latent bias:")
        print(self._b_Q.shape)
        print("Final R matrix:")
        print(self.get_complete_matrix().shape)
        print("Final RMSE:")
        print(self._training_process[self._epochs-1][1])

