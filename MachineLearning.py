import numpy as np
import random
import pickle

class Functions():
    def g(X):
        return (1/(1 + np.exp(X * -1)))
    
    def g_grad(X):
        return Functions.g(X) * np.subtract(1, Functions.g(X))

class FeatureScale():   

    def __init__(self):
        self.mean = 0
        self.std = 0
        self.type = None
        self.max = 0
        self.min = 0

    def mean_normalize(self, X):
        X = np.array(X)
        self.mean = X.mean(axis=0)
        self.max = np.amax(X, axis=0)
        self.min = np.amin(X, axis=0)          
        X = np.subtract(X, self.mean)
        X = np.divide(X, np.subtract(self.max, self.min))
        self.type = "mn"
        return X
    
    def scale(self, X):
        if self.type == "mn":
            X = np.subtract(np.array(X), self.mean)
            X = np.divide(X, np.subtract(self.max, self.min))
            return X
        elif self.type == "n":
            X = np.subtract(np.array(X), self.min)
            X = np.divide(X, np.subtract(self.max, self.min))
            return X
    
    def normalize(self, X):
        X = np.array(X)  
        self.max = np.amax(X, axis=0)
        self.min = np.amin(X, axis=0)   
        X = np.subtract(X, self.min)
        X = np.divide(X, (np.subtract(self.max, self.min)))
        self.type = "n"
        return X

class LinearRegression():
    def __init__(self) -> None:
        self.__parameters = np.array([])

    def cost(self, X, theta, Y):
        cost = (1/2) * np.mean(np.square(np.subtract(np.matmul(X, theta), Y)))
        return cost
    
    @property
    def parameters(self):
        if len(self.__parameters) == 0:
            return "The model hasn't been trained!"
        return self.__parameters

    def normal_train(self, X, Y):
        try:
            m = np.array(X).shape[0]
            X = np.hstack((np.ones((m, 1)), np.array(X)))
            Y =  np.array([Y]).transpose()
        except:
            raise Exception("Incorrect data has been passed")
        self.__parameters = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), Y)
        return self.__parameters

    def gd_train(self, X, Y, alpha=0.01):
        np.seterr(all='raise')
        try:
            m = np.array(X).shape[0]
            X = np.hstack((np.ones((m, 1)), np.array(X)))
            Y =  np.array([Y]).transpose()
        except:
            raise Exception("Incorrect data has been passed")
        if len(Y.shape) > 2:
            raise Exception("Incorrect data has been passed")
        gradient = np.zeros((X.shape[1], 1))
        theta = np.zeros((X.shape[1], 1))
        i = 0 
        while True:
            i += 1
            try:
                diff = np.subtract(np.matmul(X, theta), Y)
            except:
                raise Exception("Data values are too large, try feature scaling.")
            try:
                gradient = np.matmul(X.transpose(), diff) * (1/m)
            except:
                raise Exception("Data values are too large, try feature scaling.")
            theta = np.subtract(theta, (gradient * alpha))
            if abs(np.sum(gradient)) < 0.000001:
                break
        self.__parameters = theta
    
    def predict(self, X):
        try:
            if len(np.array(X).shape) == 1:
                X = np.array([1, *np.array(X)])
            elif len(np.array(X).shape) == 2:
                m = np.array(X).shape[0]
                X = np.hstack((np.ones((m, 1)), np.array(X)))
            else:
                raise Exception
        except:
            raise Exception("Incorrect data has been passed")
        return np.matmul(self.__parameters.transpose(), X.transpose())


class LogisticRegression():
    def __init__(self) -> None:
        self.__parameters = np.array([])

    def cost(self, X, theta, Y):
        cost = (1/2) * np.mean(np.square(np.subtract(Functions.g(np.matmul(X, theta)), Y)))
        return cost
    
    @property
    def parameters(self):
        if len(self.__parameters) == 0:
            return "The model hasn't been trained!"
        return self.__parameters

    def train(self, X, Y, alpha=0.01):
        np.seterr(all='raise')
        try:
            m = np.array(X).shape[0]
            X = np.hstack((np.ones((m, 1)), np.array(X)))
            Y =  np.array([Y]).transpose()
        except:
            raise Exception("Incorrect data has been passed")
        if len(Y.shape) > 2:
            raise Exception("Incorrect data has been passed")
        gradient = np.zeros((X.shape[1], 1))
        theta = np.zeros((X.shape[1], 1))
        i = 0 
        while True:
            i += 1
            try:
                diff = np.subtract(Functions.g(np.matmul(X, theta)), Y)
            except:
                raise Exception("Data values are too large, try feature scaling.")
            try:
                gradient = np.matmul(X.transpose(), diff) 
            except:
                raise Exception("Data values are too large, try feature scaling.")
            theta = np.subtract(theta, (gradient * alpha))
            if abs(np.sum(gradient)) < 0.0001:
                break
        self.__parameters = theta
    
    def predict(self, X):
        try:
            if len(np.array(X).shape) == 1:
                X = np.array([1, *np.array(X)])
            elif len(np.array(X).shape) == 2:
                m = np.array(X).shape[0]
                X = np.hstack((np.ones((m, 1)), np.array(X)))
            else:
                raise Exception
        except:
            raise Exception("Incorrect data has been passed")
        return Functions.g(np.matmul(self.__parameters.transpose(), X.transpose()))
    
class NeuralNetwork():
    def __init__(self):
        self.parameters = []
    
    def cost(self, X, Y, theta=None):
        if not theta:
            theta = self.parameters
        m = np.array(X).shape[0]
        if len(np.array(X).shape) == 1:
            X = np.array([X])
        m = np.array(X).shape[0]
        h0 = X
        for i in range(len(theta)):
            h0 = np.c_[np.ones((m, 1)), np.array(h0)]
            h0 = Functions.g(np.matmul(h0, theta[i]))
        J0 = np.mean(np.sum(((Y * -1) * np.log(h0)) - ((1 - Y) * np.log(1 - h0)), axis=1))
        m = np.array(X).shape[0]
        a = [np.c_[np.ones((m, 1)), np.array(X)]]
        z = []
        last_val = X
        layers = len(theta) - 1
        for i in range(layers + 1):
            m = np.array(last_val).shape[0]
            last_val = np.c_[np.ones((m, 1)), np.array(last_val)]
            last_val = np.matmul(last_val, theta[i])
            z.append(last_val)
            last_val = Functions.g(last_val)
            if i == layers:
                a.append(last_val)
            else:
                a.append(np.hstack((np.ones((m, 1)), last_val)))
        errors = []
        for i in reversed(range(layers + 1)):
            if i == layers:
                error = a[-1] - Y
            else:
                error = np.matmul(errors[0], theta[i + 1][1:,:].transpose())
                error = error * Functions.g_grad(z[i])
            errors.insert(0, error)
        gradients = []
        for i in range(len(theta)):
            gradients.append((np.matmul(a[i].transpose(), errors[i])/m))
        return J0, gradients

    
    def train(self, X, Y, nodes_in_hidden_layer, max_iter, initialized_range=[-0.3, 0.3]):
        try:
            layers = len(list(nodes_in_hidden_layer))
        except:
            raise Exception("Incorrect nodes have been passed.")
        original_parameters = []
        for i in range(layers + 1):
            if i == 0:
                previous_layer = X.shape[1]
                next_layer = nodes_in_hidden_layer[i]
            elif i == layers:
                next_layer = Y.shape[1]
                previous_layer = nodes_in_hidden_layer[i-1]
            else:
                previous_layer = nodes_in_hidden_layer[i-1]
                next_layer = nodes_in_hidden_layer[i]
            for f in range(next_layer):
                values = [random.uniform(*initialized_range) for j in range(previous_layer + 1)]
                if f == 0:
                    layer_i = np.array([values])
                else:
                    layer_i = np.concatenate((layer_i, [values]))
            original_parameters.append(layer_i.transpose())
        k = 0
        while k < max_iter:
            k += 1
            print(k)
            m = np.array(X).shape[0]
            a = [np.c_[np.ones((m, 1)), np.array(X)]]
            z = []
            last_val = X
            for i in range(layers + 1):
                m = np.array(last_val).shape[0]
                last_val = np.c_[np.ones((m, 1)), np.array(last_val)]
                last_val = np.matmul(last_val, original_parameters[i])
                z.append(last_val)
                last_val = Functions.g(last_val)
                if i == layers:
                    a.append(last_val)
                else:
                    a.append(np.hstack((np.ones((m, 1)), last_val))) 
            errors = []
            for i in reversed(range(layers + 1)):
                if i == layers:
                    error = a[-1] - Y
                else:
                    error = np.matmul(errors[0], original_parameters[i + 1][1:,:].transpose())
                    error = error * Functions.g_grad(z[i])
                errors.insert(0, error)
            gradients = []
            for i in range(len(original_parameters)):
                gradients.append((np.matmul(a[i].transpose(), errors[i])/m))
            original_parameters = [original_parameters[i] - gradients[i] for i in range(len(original_parameters))]
        self.parameters = original_parameters
        
    def predict(self, X):
        if not self.parameters:
            raise Exception("Model has not been trained.")
        m = np.array(X).shape[0]
        if len(np.array(X).shape) == 1:
            X = np.array([X])
        m = np.array(X).shape[0]
        for i in range(len(self.parameters)):
            X = np.c_[np.ones((m, 1)), np.array(X)]
            X = Functions.g(np.matmul(X, self.parameters[i]))
        return X
         

