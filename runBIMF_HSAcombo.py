# -*- coding: utf-8 -*-
'''
Created on Wed Nov 04 09:56:45 2015

@author: alex coletti
VERIFIED for Results
'''
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import FAEMED

def main():
#   A   =mpimg.imread('lena.png')
   A   =mpimg.imread('circle.png')
   A    =A[:,:,0] 
#   for RGP use  lena = lena3[:,:,0]
#   test matrix 
#   
#    A=[[8, 8, 4, 1, 5, 2, 6, 3],
#    [6, 3, 2, 3, 7, 3, 9, 3], 
#    [7, 8, 3, 2, 1, 4, 3, 7],
#    [4, 1, 2, 4, 3, 5, 7, 8],
#    [6, 4, 2, 1, 2, 5, 3, 4],
#    [1, 3, 7, 9, 9, 8, 7, 8],
#    [9, 2, 6, 7, 6, 8, 7, 7],
#    [8, 2, 1, 9, 7, 9, 1, 1]]  

   
#   N = 1000    ;     N2  = 500 
#   A = np.zeros(  (N,N)  )
#   for ii in range (1,N) :
#        for jj in range(1,N) :
#            i1 = ii - N2    
#            j1 = jj  - N2   
#            rh = np.sqrt( (i1/10.)**2 +(j1/10.)**2   )
#            A [i1+N2,j1+N2 ]  = np.sin(rh*.1*np.pi) +1.01
   nFILTER = 3
   nIMF = 1
   [B,BF]   = FAEMED.BEMD(A, nFILTER, nIMF)   
   A = BF
   if np.min(A) < 0. :
        f_delta  =  (np.max (A)- np.min (A) ) 
        A       = A   +   1.01 *  f_delta
   [Ph, A_c]= FAEMED.HSA(A) 
   [N1,N2] = np.shape(A)   

#   x =  np.arange(0, N1-1)
#   plt.plot(x, A[N1/2,x])
#    plt.show(x,y)
#    plt.gray()
   plt.imshow(Ph,cmap=plt.get_cmap('hot') , vmin=0, vmax=1)
#   plt.imshow(Ph )
main()