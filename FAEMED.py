# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 18:31:56 2015
@author: acoletti
"""
import numpy as np
def BEMD(B, Wen, stop):
    # Function computes the Bi dimensional decompsition
    # needed by the Hilber Huang Transfrom and Spectral Analysis of one image
    # The Empirical Mode Decompostion EMD is perfromed according to:
    #Kizhner, S.; Blank, K.B.; Sichler, J.A.; Patel, U.D.; Le Moigne, J.; El-Araby, E.; Vinh Dang,
    #"On development of Hilbert-Huang Transform data processing real-time system with 2D capabilities," 
    #in Adaptive Hardware and Systems (AHS), 2012 NASA/ESA Conference on , vol., no., pp.233-238, 25-28 June 2012
    #doi: 10.1109/AHS.2012.6268656
    #URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6268656&isnumber=6268630
    
    # Code Author: Alex COletti
    # Code Date: November 4, 2015
    #==========================================================
    # INPUTS:
    # B      = 2 dimensional matrix
    # Wen    = window size of the EMD
    # stop   = Number of iterations to compute
    # OUTPUTS
    # A      = Residual order "stop"
    # BIMF1  = Bi-  Intrinsic Monde Function of Order "stop"
    #==========================================================
    # Define default values
    A = np.float64(B)   
    [N1,N2] =np.shape(A)      
    k = int((Wen-1)/2 )   
    ii   =   jj  = k + 1     #      jj=k+1
    for dstop in range (stop):
        Uen   =   np.zeros( (N1, N2) )
        Len   =   np.zeros( (N1, N2) )
        npad = ((k,k),(k,k))
        M_imb = np.pad(A, npad,mode='constant',constant_values=0.)
        M     =   np.zeros( ( (2*k + 1) , (2*k+1) )  )
        for ii in range     ((k),(N1 + k)):   
            for jj in range ((k),(N2 + k)):
                M = M_imb[ (ii-k):(ii + k+1) ,   (jj-k) : (jj + k+1) ]  
                Uen[ii-k, jj-k] = np.max(M) # Ok for MAX because 0 will not be a max
                if (np.count_nonzero(M)==0) :
                    Len[ii-k, jj-k]=0
                else:        
                    Len[ii-k, jj-k] = np.min(M[np.nonzero(M)]) # The above line MAX doesn't work for MIN   
#   =====================================================================
    M_imb  = np.pad(Uen, npad,mode='constant',constant_values=0.)
    M_imb2 = np.pad(Len, npad,mode='constant',constant_values=0.)    
    M     =  np.zeros( ( (2*k + 1) , (2*k+1) )  )
    Uensmooth = np.zeros( (N1, N2) ) #Initialise an NxN array to zeros
    Lensmooth = np.zeros( (N1, N2) ) #Initialise an NxN array to zeros
    for ii in range     ((k),(N1 + k)):
        for jj in range ((k),(N2 + k)):
            M = M_imb[  (ii-k):(ii + k+1) , (jj-k):(jj + k+1)  ]
            count1 = np.count_nonzero(M)
            count0 = count1==0
            if (count0) :
                Uensmooth[ii-k, jj-k]=0
                test = [ii,jj,count1, Uensmooth[ii-k, jj-k]]
            else:
                Uensmooth[ii-k, jj-k] = np.sum(np.sum(M)) / np.count_nonzero(M) #0's don't matter here
                test = [ii,jj,count1, Uensmooth[ii-k, jj-k]]
            M = M_imb2[  (ii-k):(ii + k+1) , (jj-k):(jj + k+1)  ]
            count1 = np.count_nonzero(M)
            count0 = count1==0
            if (count0) :
                Lensmooth[ii-k, jj-k]=0
            else:
                Lensmooth[ii-k, jj-k] = np.sum(M[np.nonzero(M)]) / np.count_nonzero(M)
#    print('Uensmooth')   
#    print (Uensmooth)
#    print('Lensmooth') 
#    print (Lensmooth)


    Median = (Uensmooth + Lensmooth)/2.0
    BIMF1 = A - Median #071411 by SK
    Res1=A-BIMF1  
    A=Res1
    ii   =   jj=   k     #      jj=k+1
    return A, BIMF1 
'''
Created on Wed Nov 04 09:56:45 2015

@author: alex coletti
VERIFIED for Results
'''
from scipy.signal import hilbert as hilb
def HSA (B)  :
    '''
    -------------------------------------------------------------------
    Transaltion MATLAB to Python
    http://mathesaurus.sourceforge.net/matlab-numpy.html
    ===================================================================
    Function HSA 
    Performs the Hilbert Transform of an imput matrix
    or image in 2D
    Lihong Qiao, Sisi Chen
    "Two Dimensional Hilbert Spectral Characters Analysis Based on
    Bi-orthant Analytic Signal"
    International Journal of Digital Content Technology and its 
    Applications(JDCTA)
    Volume5,Number9,September 2011
    doi:10.4156/jdcta.vol5.issue9.34
    ====================================================================
    Function 
    INPUT  
    A  = input matrix in 2D
    OUTPUTS
    I_Phase = Phase 
    ASfxy   = Complex matrix where 
          real ASfxy = input correct to avoid divisions by zero
          imaginary  = BHT  Complex Amplitude
    =====================================================================
    November 25, 2015: 
    Function verified against NASA GSFC version 
    by Semion Kizhner
    This function in MarLab uses the function hilb 
    By default executes by column unless axis # is specified
   '''
    A       =  np.float64(B)
    [N1,N2] = np.shape(A)    
    f_min   =  (np.max (A)- np.min (A) )/10000.    
    fxy     = A + f_min 
    '''
    ** Cleaner for Hilbert function by rows!
    An suggests this version for an Hilber that process by rows (index=0)  
    Hx(of fxy) = hilb2(fxy).imag
    Hy(of fxy) = (hilb2( (fxy). T )).imag.T
    Ht(of fxy) = Hy(of Hx) = (hilb2( (Hx).T )).imag.T
    Hx(of Ht) = hilb2(Ht).imag
    Hy(of Ht) = (hilb2( (Ht). T )).imag.T
    Ht(of Ht) = Hy(of Hx(of Ht)) = (hilb2( (hilb2(Ht).imag). T )).imag.T
    BHT = Hx + (hilb2(  (fxy –  ((hilb2( (hilb2(Ht).imag). T )).imag.T)  ).T)).imag.T
    BHT = Hx + (hilb2(  (fxy –  ((hilb2( (Hx(of Ht). T )).imag.T)  ).T)).imag.T
    **** According to documentation Hilber2 
    scipy.signal.hilbert2(x, N=None)[source]¶
    Parameters:	
        x : array_like
        2-D signal data.
        N : int or tuple of two ints, optional
        Number of Fourier components. Default is x.shape
        Returns:	
        xa : ndarray
        Analytic signal of x taken along axes (0,1)

    --------------
     scipy.signal.hilbert(x, N=None, axis=-1)[source]
     Compute the analytic signal, using the Hilbert transform.
     The transformation is done along the last axis by default.
     Parameters:	
         x : array_like
         Signal data. Must be real.
         N : int, optional
         Number of Fourier components. Default: x.shape[axis]
         axis : int, optional
         Axis along which to do the transformation. Default: -1.
         Returns:	
         xa : ndarray
         Analytic signal of x, of each 1-D array along axis    
    '''    
    Hx     =  hilb  (fxy, axis =0 ).imag
#    Hy     = (hilb  (fxy) ).imag   # not in use
    Ht     = (hilb  (Hx ) ).imag   # Ht = (hilb2( (Hx).T   )).imag.T
    Hx_of_Ht =  hilb  (Ht, axis = 0).imag
#    Hy_of_Ht =  hilb  (Ht          ).imag  # not in use
#    Ht_of_Ht =  hilb  (Hx_of_Ht    ).imag  # not in use
    BHTint   = fxy- hilb(Hx_of_Ht, axis =0).imag  # fxy – hilb(Hx_of_Ht, axis =0).imag
    BHT      =  Hx + (hilb(BHTint,axis=0).imag.T)
    ASfxy  = fxy + BHT*1j
    I_Phase = np.arctan(np.divide(BHT, fxy))
    return (I_Phase,  ASfxy)

