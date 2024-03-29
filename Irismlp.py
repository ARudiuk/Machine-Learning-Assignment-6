import numpy as np

#normalize data
#lots of code in this file is copied from the book
iris = np.loadtxt('iris_proc.dat',delimiter=',')
iris[:,:4] = iris[:,:4]-iris[:,:4].mean(axis=0)
imax = np.concatenate((iris.max(axis=0)*np.ones((1,5)),np.abs(iris.min(axis=0)*np.ones((1,5)))),axis=0).max(axis=0)
iris[:,:4] = iris[:,:4]/imax[:4]

# Set up target data as 1-of-N output encoding
target = np.zeros((np.shape(iris)[0],3));
indices = np.where(iris[:,4]==0) 
target[indices,0] = 1
indices = np.where(iris[:,4]==1)
target[indices,1] = 1
indices = np.where(iris[:,4]==2)
target[indices,2] = 1

# Randomly order the data so that we get a division of classes when we split the data
order = range(np.shape(iris)[0])
np.random.shuffle(order)
iris = iris[order,:]
target = target[order,:]

#impot ANN
import ANN
sum = 0.0
for i in range(100):
    #initialize perceptron
    if i%10 == 0:
        print i
    net = ANN.ANN(iris[:,:4],target,nhidden1 = 5,nlayers = 1,momentum = 0.4 )
    #split the data we randomized and encoded the output for earlier
    net.split_50_25_25()
    #train for n iterations
    #first parameter is number of iterations
    #second parameter is the learning rate
    #third parameter is a boolean of whether or not you want to track and plot the error during training
    points,max_value = net.train_n_iterations(2000,0.3,plot_errors = False)
    #print confusion matrix
    net.confmat(net.valid,net.validt,print_info = False)
    sum+=net.confmat(net.test,net.testt,print_info = False)
print sum/100.0