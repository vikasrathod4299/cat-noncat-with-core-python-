import numpy as np 
import matplotlib.pyplot as plt
import h5py
import scipy
import skimage
import tensorflow as tf
from skimage import transform
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
from scipy import ndimage
from lr_utils2 import *

np.random.seed(0)


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

#flatining  
train_x_flt = train_x_orig.reshape(train_x_orig.shape[0],-1).T
test_x_flt= test_x_orig.reshape(test_x_orig.shape[0],-1).T 

#normalizing inputs
train_x=train_x_flt/255.
test_x=test_x_flt/255.

layer_dims=[train_x.shape[0], 20,7,5, 1]
m=train_x.shape[1]
m2=test_y.shape[1]


def neural_network(X,Y,layer_dims,learning_rate=0.03,num_iteration=500,print_cost=False):

	np.random.seed(1)
	costs = []
	parameters = initialize_parameters_deep(layer_dims)

	for i in range(0,num_iteration):


		AL,caches=L_model_forward(X,parameters,0.5)
		cost = compute_cost(AL,Y,parameters,0.1) 
		
		grads=L_model_backward(AL,Y,caches,parameters,0.1)
		

		parameter=update_parameters(parameters,grads,learning_rate)

		if print_cost and i % 100 == 0:
			print("cost after itrations %i: %f"%(i,cost))
		if print_cost and i % 100 == 0:
			costs.append(cost) 
            
            
	plt.plot(np.squeeze(costs))
	plt.xlabel('Cost')
	plt.ylabel('itrations')
	plt.title('learning_rate = '+str(learning_rate))
	plt.show()

	return parameters

parameters=neural_network(train_x,train_y,layer_dims,learning_rate=0.03,num_iteration=2000,print_cost=True)


predictios= predict(train_x,parameters)
predictios_test= predict(test_x,parameters)
print(predictios_test)
print(test_y)
print("\nTrain set accuracy: "  + str((np.sum((predictios == train_y)/m))*100))
print("Tst set accuracy: "  + str(np.sum((predictios_test == test_y)/m2)*100))
