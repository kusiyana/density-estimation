#!/usr/bin/python

# -----------------------------------------------------------------------------
# This is a small script to test Python's ability to use a density estimation
# classifier to "learn" binary classification of a new data point given a
# gaussian fit of known points with known classifications. Adapted from Matlab
# script by David Barber
#
# Hayden Eastwood, September 2013
# -----------------------------------------------------------------------------

import sys
# add package system path if need be
sys.path.append('/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages')

import numpy as np
import matplotlib.pyplot as plt
import math

# set up data set 1
x1 = np.random.rand(2,10)
x2 = np.random.rand(2,10)

# fit gaussian
m1 = np.mean(x1, axis=1)
S1 = np.cov(x1)
invS1 = np.matrix(S1).I
det1 = np.linalg.det(S1)

p1 = float(x1.shape[1])/(x2.shape[1] + x1.shape[1]) # prior

# set up data set 2
x2 = np.kron(np.ones((1,15)), np.array([[0.6],[0.8]])) + np.random.rand(2,15)

# fit gaussian
m2 = np.mean(x2, axis=1)
S2 = np.cov(x2)
invS2 = np.matrix(S2).I
det2 = np.linalg.det(S2)
p2 = 1 - p1

# perform classification
xNew = np.kron(np.ones((1,4)), np.array([[0.3],[0.4]])) + np.random.rand(2,4)
d1 = xNew.T - np.kron(np.ones((xNew.shape[1],1)),m1)
d2 = xNew.T - np.kron(np.ones((xNew.shape[1],1)),m2)

d1 = d1.T
d2 = d2.T

clas = np.zeros((1,xNew.shape[1])) #initialise classification matrix
for i in range(xNew.shape[1]):
	if np.inner(np.inner(d2[:,i].T, invS2), d2[:,i]) + det2 - 2 * np.log(p2) > np.inner(np.inner(d1[:,i].T, invS1), d1[:,i]) + det1 - 2 * np.log(p1):
		print(np.inner(np.inner(d2[:,0].T, invS2), d2[:,0]) + det2 - 2 * np.log(p2))
		clas[0,i] = 2
		 

#show plots
plt.plot(x2[0,:], x2[1,:], 'o')
plt.plot(x1[0,:], x1[1,:], 'x')
plt.plot(xNew[0,:], xNew[1,:], 'yo')
plt.show()
