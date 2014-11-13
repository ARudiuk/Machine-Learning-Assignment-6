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

print "Train size is", np.shape(train)[0]

import som
net = som.som(12,12,train)
net.somtrain(train,400)

best = np.zeros(np.shape(train)[0],dtype=int)
count = np.zeros((12*12,))+15
for i in range(np.shape(train)[0]):
    best[i],activation = net.somfwd(train[i,:])
    count[best[i]] = count[best[i]]+15

pl.plot(net.map[0,:],net.map[1,:],'k.',ms=15)
where = pl.find(traint == 0)
print count
for i in range(np.shape(where)[0]):
    temp1 = best[where[i]]
    temp2 = count[temp1]
    pl.plot(net.map[0,temp1],net.map[1,temp1],'rs',ms=temp2)
where = pl.find(traint == 1)
for i in range(np.shape(where)[0]):
    temp1 = best[where[i]]
    temp2 = count[temp1]
    pl.plot(net.map[0,temp1],net.map[1,temp1],'gv',ms=temp2)
where = pl.find(traint == 2)
for i in range(np.shape(where)[0]):
    temp1 = best[where[i]]
    temp2 = count[temp1]
    pl.plot(net.map[0,temp1],net.map[1,temp1],'b^',ms=temp2)
pl.title("Train Data")
pl.axis([-0.1,1.1,-0.1,1.1])
pl.axis('off')
pl.figure(2)

best = np.zeros(np.shape(test)[0],dtype=int)
count = np.zeros((12*12,))+15
for i in range(np.shape(test)[0]):
    best[i],activation = net.somfwd(test[i,:])
    count[best[i]] = count[best[i]]+15

pl.plot(net.map[0,:],net.map[1,:],'k.',ms=15)
where = pl.find(testt == 0)
print count[where]
for i in range(np.shape(where)[0]):
    temp1 = best[where[i]]
    temp2 = count[temp1]
    pl.plot(net.map[0,temp1],net.map[1,temp1],'rs',ms=temp2)
where = pl.find(testt == 1)
print count[where]
for i in range(np.shape(where)[0]):
    temp1 = best[where[i]]
    temp2 = count[temp1]
    pl.plot(net.map[0,temp1],net.map[1,temp1],'gv',ms=temp2)
where = pl.find(testt == 2)
print count[where]
for i in range(np.shape(where)[0]):
    temp1 = best[where[i]]
    temp2 = count[temp1]
    pl.plot(net.map[0,temp1],net.map[1,temp1],'b^',ms=temp2)
pl.title("Test Data")
pl.axis([-0.1,1.1,-0.1,1.1])
pl.axis('off')
pl.show()
