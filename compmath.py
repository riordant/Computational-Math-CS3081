import scipy
import numpy as np
import math
import scipy.misc
from numpy.linalg import inv
from matplotlib import pyplot as plt
#from pylab import *
%pylab inline
 
10S = [0.038,0.194,.425,.626,1.253,2.500,3.740]
rates = [0.050,0.127,0.094,0.2122,0.2729,0.2665,0.3317]
iterations = 5 # max iterations
rows = 7 #no. of data points
cols = 2 #no. of unknowns in model function
 
B = np.matrix([[.9],[.2]]) # original guess for unknowns
Jr = np.zeros((rows,cols)) # Jacobian matrix from r
r = np.zeros((rows,1)) #r equations
 
20def model(Vmax, Km, Sval): # model function
   return ((Vmax * Sval) / (Km + Sval))
   
def partialDerB1(B2,xi): # partial derivative of residual with respect to B1
   return round(-(xi/(B2+xi)),10)
 
def partialDerB2(B1,B2,xi): # partial derivative of residual with respect to B2
   return round(((B1*xi)/((B2+xi)*(B2+xi))),10)
   
def residual(x,y,B1,B2): # residual function
30   return (y - ((B1*x)/(B2+x)))
 
 
for i in range(0,iterations): Gauss Newton. iterate "iterations" times
 
   sumOfResid=0
   #calculate Jr,r and the sum of residuals for this iteration.
   for j in range(0,rows):
      r[j,0] = residual(S[j],rates[j],B[0],B[1])
      sumOfResid = sumOfResid + (r[j,0] * r[j,0])
40      Jr[j,0] = partialDerB1(B[1],S[j])
      Jr[j,1] = partialDerB2(B[0],B[1],S[j])
   
   Jrt =  np.transpose(Jr) # transposition of
   
   B = B - np.dot(np.dot(inv(np.dot(Jrt,Jr)),Jrt),r) # calc next Beta vector.   B(next) = B(this) - inv(Jr^T*Jr)Jr^Tr(B(this)
   
   print "sum of the squares of the residuals after iteration",(i+1), ":"  
   print sumOfResid
   print "Current Beta:"
50   print B
#contruct y values for curve from original model function 
curveRate = [0]*rows
for i in range(0,rows):
        curveRate[i] = model(B[0,0],B[1,0],S[i])
#plot output
plt.xlabel("[S]")
plt.ylabel("rate of reaction")
plt.plot(S,rates)
plt.plot(S,curveRate)
