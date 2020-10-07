import numpy as np
import h5py	


def load_data():
	train_dataset = h5py.File('C:/Users/Intel/Desktop/python/Logistic reagression/train_catvnoncat.h5', "r")
	train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
	train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
	test_dataset = h5py.File('C:/Users/Intel/Desktop/python/Logistic reagression/test_catvnoncat.h5', "r")
	test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
	test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
	classes = np.array(test_dataset["list_classes"][:]) # the list of classes
	train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
	test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_parameters_deep(layer_dims):
	
	np.random.seed(1)
	L = len(layer_dims)
	parameters = {}

	for l in range(1, L):
		parameters['W' 	+ str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
		parameters['b' + str(l)] = np.zeros(shape=(layer_dims[l], 1)) 

	assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
	assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
	
	return parameters

def lin_forward(A_prev, W, b):
	
	Z = np.dot(W,A_prev) + b
	cache = (A_prev, W, b)
	
	assert(Z.shape == (W.shape[0], A_prev.shape[1]))

	return Z, cache


def sigmoid(Z):

	A=1/(1+np.exp(-Z))
	cache=Z

	return A,cache

def relu(Z):

	A=np.maximum(0,Z)
	cache=Z

	return A, cache


def lin_act_forward(A_prev,W,b,activation):

	if activation == "sigmoid":
		Z,linear_cache=lin_forward(A_prev,W,b)
		A,act_cache=sigmoid(Z)

	elif activation =="relu":
		Z,linear_cache=lin_forward(A_prev,W,b)
		A,act_cache=relu(Z)
	
	cache=(linear_cache,act_cache)
	
	assert (A.shape == (W.shape[0], A_prev.shape[1]))
	
	return A,cache


def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                  
    
    for l in range(1, L):
        A_prev = A 

        A, cache = lin_act_forward(A_prev, 
                                             parameters['W' + str(l)], 
                                             parameters['b' + str(l)], 
                                             activation='relu')
        caches.append(cache)

    AL, cache = lin_act_forward(A, 
                                          parameters['W' + str(L)], 
                                          parameters['b' + str(L)], 
                                          activation='sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches


def compute_cost(AL, Y):
   
    m = Y.shape[1]

    cost = (-1/m)* np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))
    cost = np.squeeze(cost)  
    
    assert(cost.shape == ())
    
    return cost

def lin_backward(dZ,cache):

	A_prev,W,b= cache
	m=A_prev.shape[1]

	dW=(1./m)*np.dot(dZ,A_prev.T)
	db=(1./m)*np.sum(dZ,axis=1,keepdims=True)
	dA_prev=np.dot(W.T,dZ)

	assert(dA_prev.shape == A_prev.shape)
	assert(dW.shape == W.shape)
	assert(db.shape == b.shape)

	return dA_prev,dW,db 


def sigmoid_backward(dA,cache):
	Z=cache
	s=1/(1+np.exp(-Z))
	dZ=dA*s*(1-s)
	assert (dZ.shape == Z.shape)
	return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ


def act_backward(dA,cache,activation):
	linear_cache,activation_cache=cache

	if activation=="relu":
		dZ=relu_backward(dA,activation_cache)
		dA_prev,dW,db=lin_backward(dZ,linear_cache)
	elif activation == "sigmoid":
		dZ=sigmoid_backward(dA,activation_cache)
		dA_prev,dW,db=lin_backward(dZ,linear_cache)

	return dA_prev, dW, db

def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = act_backward(dAL, current_cache, activation = "sigmoid")
   
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = act_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters,grads,learning_rate):
	
	L=len(parameters)//2

	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

	return parameters

def predict(X,parameters):

    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))

    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    

   
        
    return p