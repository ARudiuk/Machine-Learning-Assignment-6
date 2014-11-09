import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
#normalize data
iris = np.loadtxt('iris_proc.dat',delimiter=',')[:,:4]
iris /= np.max(np.abs(iris),axis=0)

import kmeansnet
ann = kmeansnet.kmeans(3,iris,nEpochs=1000)
ann.kmeanstrain(iris)
print ann.kmeansfwd(iris)