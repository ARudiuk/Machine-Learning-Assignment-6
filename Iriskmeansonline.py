import pylab as pl
import numpy as np

iris = np.loadtxt('iris_proc.dat',delimiter=',')
iris[:,:4] = iris[:,:4]-iris[:,:4].mean(axis=0)
imax = np.concatenate((iris.max(axis=0)*np.ones((1,5)),iris.min(axis=0)*np.ones((1,5))),axis=0).max(axis=0)
iris[:,:4] = iris[:,:4]/imax[:4]

target = iris[:,4]

order = range(np.shape(iris)[0])
np.random.shuffle(order)
iris = iris[order,:]
target = target[order]

train = iris[::2,0:4]
traint = target[::2]
valid = iris[1::4,0:4]
validt = target[1::4]
test = iris[3::4,0:4]
testt = target[3::4]

#print train.max(axis=0), train.min(axis=0)

import kmeansnet
#import kmeans as kmeansnet
net = kmeansnet.kmeans(3,train)
net.kmeanstrain(train)
cluster = net.kmeansfwd(test)
print 1.*cluster
print iris[3::4,4]