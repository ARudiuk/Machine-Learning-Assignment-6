import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
#normalize data
iris = np.loadtxt('iris_proc.dat',delimiter=',')[:,:4]
iris /= np.max(np.abs(iris),axis=0)

# print iris

#impot ANN
import ANN
#initialize perceptron
net = ANN.ANN(iris,iris,nhidden1 = 3,nlayers = 1,momentum = 0)
#split the data we randomized and encoded the output for earlier
#train for n iterations
#first parameter is number of iterations
#second parameter is the learning rate
#third parameter is a boolean of whether or not you want to track and plot the error during training
net.train_n_iterations(5000,0.3,plot_errors = False)
net.test_associative(net.train, iris)
#count number of each compressed node
#as defined by the node with the largest activation
distribution = [0, 0, 0]
plot_data = net.hidden1[:,1:]
for i in net.hidden1:
    # print i[1:]
    distribution[np.argmax(i[1:])] += 1

print distribution
print np.argmax(net.hidden1[:,1:],axis=1)

fig = pl.figure()
ax = Axes3D(fig)
ax.scatter(plot_data[:50,0],plot_data[:50,1],plot_data[:50,2],c='r',marker = 'x')
ax.scatter(plot_data[50:100,0],plot_data[50:100,1],plot_data[50:100,2],c='b',marker = 'o')
ax.scatter(plot_data[100:150,0],plot_data[100:150,1],plot_data[100:150,2],c='g',marker = 'v')
ax.set_xlabel('A1')
ax.set_ylabel('A2')
ax.set_zlabel('A3')
fig.add_axes(ax)
pl.show()