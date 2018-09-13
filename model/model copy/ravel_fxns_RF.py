import numpy as np
import pandas as pd
import scipy as sp
import scipy.ndimage
from scipy import ndimage

def RF_patterns(isveg, rvl_params):        
       
    """
    identical to local_patterns, but no patterns.pklz files will be saved.
  
    inputs:
      ncol, nrow, isveg 
  
    output: pattern_dict 
      d2uB (ncol x nrow) : distance to nearest upslope bare cell
      d2dB (ncol x nrow) : distance to nearest downslope bare cell  
      d2yB (ncol x nrow) : distance to nearest along-slope bare cell   
      d2lB (ncol x nrow) : distance to nearest left bare cell
      d2rB (ncol x nrow) : distance to nearest right bare cell  
      d2xB (ncol x nrow) : distance to nearest across-slope bare cell      
    
    
    omitted: 
     d2divide (ncol x nrow) : distance to divide
     d2div (nrow) : distance to divide
  
    """
    isvegc = np.array(isveg, dtype = float) 
    
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    
    edge = int(rvl_params['edge'])    
    saturate = int(rvl_params['saturate'])        
    weight = int(rvl_params['weight'])            
    # gsigma = int(rvl_params['gsigma'])
        
    d2wB = func_d2wB(isvegc, saturate, weight)      
    d2B = func_d2B(isvegc, saturate, weight)          

    d2uB = func_d2uB(isvegc, edge ,saturate)
    d2dB = func_d2dB(isvegc, edge ,saturate)     
        
    d2yB = ndimage.distance_transform_edt(isvegc, sampling = (10, 1))
    d2yB[d2yB > saturate] = saturate
    
    d2lB = func_d2lB(isvegc, edge, saturate)
    d2rB = func_d2rB(isvegc, edge, saturate)    
    #d2xB = np.fmin(d2lB, d2rB)
    d2xB =  ndimage.distance_transform_edt(isvegc, sampling = (1, 10))
    d2xB[d2xB > saturate] = saturate
        
    d2uV = func_d2uV(isvegc, edge, saturate)
    d2dV = func_d2dV(isvegc, edge, saturate)
    d2yV = np.fmin(d2uV, d2dV)
    
    d2lV = func_d2lV(isvegc, edge, saturate)
    d2rV = func_d2rV(isvegc, edge, saturate)   
    d2xV = np.fmin(d2lV, d2rV)
          
    patchLv,patchLb = get_patchL(isvegc, edge, saturate) 
    #upslope_sum = np.flip(np.cumsum(np.flip(isvegc, 1), 1), 1)
    upslope10 = upslope_memory(isvegc, edge, min(nrow, 10))
    upslope3 = upslope_memory(isvegc, edge, min(nrow, 3))    
    
    pattern_dict = {'isvegc' : isvegc,
                    'd2wB' : d2wB, 
                    'd2B' : d2B,                     
                    'd2uB' : d2uB, 
                    'd2dB' : d2dB, 
                    'd2yB' : d2yB,
                    'd2lB' : d2lB, 
                    'd2rB' : d2rB, 
                    'd2xB' : d2xB,
                    'd2uV' : d2uV, 
                    'd2dV' : d2dV, 
                    'd2yV' : d2yV,
                    'd2lV' : d2lV, 
                    'd2rV' : d2rV, 
                    'd2xV' : d2xV,                    
                    'patchLv' : patchLv, 
                    'patchLb' : patchLb,                    
                    'upslope10' : upslope10,
                    'upslope3' : upslope3       
                  }
                  
    for key in ['d2uV','d2xV','d2yV' ]:
      pattern_dict[key + '_s'] =   smoothB(pattern_dict[key], isvegc, 2)        
    
    for key in ['d2uB','d2xB','d2yB', 'patchLb']:
      pattern_dict[key + '_s'] =   smoothV(pattern_dict[key], isvegc, 2)        
            
    return pattern_dict                

            

def smoothB(U, isvegc, gsigma):
    U = U.astype(float)
    U[isvegc == 1] = np.nan
    V=U.copy()
    V[U!=U]=0
    VV=sp.ndimage.gaussian_filter(V,gsigma)

    W=0*U.copy()+1
    W[U!=U]=0
    WW=sp.ndimage.gaussian_filter(W,gsigma)

    Z=VV/WW
    Z = Z.astype(int)
    Z[isvegc ==1] = 0
    return Z

            

def smoothV(U, isvegc, gsigma):
    U = U.astype(float)
    U[isvegc == 0] = np.nan
    V=U.copy()
    V[U!=U]=0
    VV=sp.ndimage.gaussian_filter(V,gsigma)

    W=0*U.copy()+1
    W[U!=U]=0
    WW=sp.ndimage.gaussian_filter(W,gsigma)

    Z=VV/WW
    Z = Z.astype(int)
    Z[isvegc ==0] = 0
    return Z

    
    
def func_d2wB(isvegc, saturate, weight):
    """
    Distane to weighted nearest upslope bare cell
    =  0 for bare ground
    =  1 for veg cells with a neighboring bare cell upslope
    >  1 for veg cells with bare cells further upslope
    """
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    
    res =  isvegc.copy()
    
    for i in range(nrow):
        d = isvegc.copy()
        d[:,:i+1] = 1
        res[:,i] = ndimage.distance_transform_edt(d, sampling = (weight, 1))[:, i]   
    res[isvegc ==0] = 0
    
    res[res>saturate] = saturate   
    return res


def func_d2B(isvegc, saturate, weight):
    """
    Distane to weighted nearest upslope bare cell
    =  0 for bare ground
    =  1 for veg cells with a neighboring bare cell upslope
    >  1 for veg cells with bare cells further upslope
    """
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    
    res =  isvegc.copy()
    
    for i in range(nrow):
        d = isvegc.copy()
        d[:,:i+1] = 1
        res[:,i] = ndimage.distance_transform_edt(d, sampling = (1, 1))[:, i]   
    res[isvegc ==0] = 0

    res[res>saturate] = saturate    
    
    return res

    
def func_d2uB(isvegc, edge, saturate):
    """
    Distane to nearest upslope bare cell
    =  0 for bare ground
    =  1 for veg cells with a neighboring bare cell upslope
    >  1 for veg cells with bare cells further upslope
    """

    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    arr = np.ones([ncol+ 2*edge, nrow + 2*edge]).T
    arr[edge:-edge, edge:-edge] = np.flipud(isvegc.T)
    a = pd.DataFrame(arr) != 0

    df1 = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0).astype(int)
    df1 = np.array(df1)[edge:-edge, edge:-edge]
    df1 = np.flipud(df1).T
    df1[isvegc == 0] = 0
   
    df1[df1>saturate] = saturate
    
    return df1
      
# def func_d2uB(isvegc, edge):
#     """
#     Distane to nearest upslope bare cell
#     =  0 for bare ground
#     =  1 for veg cells with a neighboring bare cell upslope
#     >  1 for veg cells with bare cells further upslope
#     """
#     d2uB = np.zeros_like(isvegc, dtype = float)
#
#     ncol = isvegc.shape[0]
#     nrow = isvegc.shape[1]
#
#     for j in range(ncol):
#       for k in range(nrow):
#         if isvegc[j,k] == 1:
#           i = 1
#           stopFlag = 0
#           while stopFlag == 0:
#             try:
#               if isvegc[j, k+i] == 0:
#                 d2uB[j,k] = i
#                 stopFlag = 1
#               else:
#                 i = i+1
#             except IndexError:
#               d2uB[j,k] = i + edge
#               stopFlag = 1
#
#     return d2uB
    
def func_d2uV(isvegc, edge, saturate):
    """
    Distane to nearest upslope veg cell
    =  0 for veg cells
    =  1 for bare cells with a neighboring veg cell upslope
    >  1 for bare cells with a  veg cell further upslope
    """
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    arr = np.zeros([ncol+ 2*edge, nrow + 2*edge]).T
    arr[edge:-edge, edge:-edge] = 1- np.flipud(isvegc.T)
    a = pd.DataFrame(arr) != 0

    df1 = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0).astype(int)
    df1 = np.array(df1)[edge:-edge, edge:-edge]
    df1 = np.flipud(df1).T
    df1[isvegc == 1] = 0

    df1[df1>saturate] = saturate  
    return df1
    # d2uV = np.zeros_like(isvegc, dtype = float)
    #
    # ncol = isvegc.shape[0]
    # nrow = isvegc.shape[1]
    #
    # for j in range(ncol):
    #     for k in range(nrow):
    #         if isvegc[j,k] == 0:
    #             i = 1
    #             stopFlag = 0
    #             while stopFlag == 0:
    #                 try:
    #                     if isvegc[j, k+i] == 1:
    #                         d2uV[j,k] = i
    #                         stopFlag = 1
    #                     else:
    #                         i = i+1
    #                 except IndexError:
    #                         d2uV[j,k] = i + edge
    #                         stopFlag = 1

    
        

def func_d2dB(isvegc, edge, saturate):
    """
    input: 
      isvegc : [ncol x nrow] array of vegetation field
  
    output : 
      d2dB : [ncol x nrow] array with distance to nearest downslope bare cell
        =  0 for bare ground
        =  1 for veg cells with a bare cell immediately downslope
    """    
    
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    
    arr = np.ones([ncol+ 2*edge, nrow + 2*edge]).T
    arr[edge:-edge, edge:-edge] = isvegc.T
    a = pd.DataFrame(arr) != 0

    df1 = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0).astype(int)
    df1 = np.array(df1)[edge:-edge, edge:-edge]
    df1 = df1.T
    df1[isvegc == 0] = 0    
     
    df1[df1>saturate] = saturate
 
    return df1

    
# def func_d2dB(isvegc, edge):
#     """
#     input:
#       isvegc : [ncol x nrow] array of vegetation field
#
#     output :
#       d2dB : [ncol x nrow] array with distance to nearest downslope bare cell
#         =  0 for bare ground
#         =  1 for veg cells with a bare cell immediately downslope
#     """
#     d2dB = np.zeros_like(isvegc, dtype = float)
#
#     ncol = isvegc.shape[0]
#     nrow = isvegc.shape[1]
#
#     for j in range(ncol):
#       for k in range(nrow-1,-1,-1):
#         if isvegc[j,k] == 1:
#           i = 1
#           stopFlag = 0
#           while stopFlag == 0:
#               try:
#                   if  k-i < 0:
#                     d2dB[j,k] = i+2
#                     stopFlag = 1
#                   if isvegc[j, k-i] == 0:
#                       d2dB[j,k] = i
#                       stopFlag = 1
#                   else:
#                       i = i+1
#               except IndexError:
#                       d2dB[j,k] = i + edge
#                       stopFlag = 1
#
#     return d2dB


def func_d2dV(isvegc, edge, saturate):
    """
    input: 
      isvegc : [ncol x nrow] array of vegetation field

    output : 
      d2dV : [ncol x nrow] array of distane to nearest downslope veg cell
        =  0 for veg cells
        =  1 for bare cells with a bare cell immediately downslope  
    """
  

    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    arr = np.zeros([ncol+ 2*edge, nrow + 2*edge]).T
    arr[edge:-edge, edge:-edge] = 1 - isvegc.T
    a = pd.DataFrame(arr) != 0

    df1 = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0).astype(int)
    df1 = np.array(df1)[edge:-edge, edge:-edge]
    df1 =  df1.T
    df1[isvegc == 1] = 0
   
    df1[df1>saturate] = saturate
    
    return df1

   # d2dV = np.zeros_like(isvegc, dtype = float)
   #  ncol = isvegc.shape[0]
   #  nrow = isvegc.shape[1]
   #  for j in range(ncol):
   #      for k in range(nrow-1,-1,-1):
   #          if isvegc[j,k] == 0:
   #              i = 1
   #              stopFlag = 0
   #              while stopFlag == 0:
   #                  try:
   #                      if  k-i < 0:
   #                          d2dV[j,k] = i+2
   #                          stopFlag = 1
   #                      elif isvegc[j, k-i] == 1:
   #                          d2dV[j,k] = i
   #                          stopFlag = 1
   #                      else:
   #                          i = i+1
   #                  except IndexError:
   #                          print j,k
   #                          d2dV[j,k] = i + edge
   #                          stopFlag = 1




def func_d2lB(isvegc, edge, saturate):
    """
    input:
      isvegc : [ncol x nrow] array of vegetation field

    output :
      d2lB : [ncol x nrow] array of distane to nearest left bare
        =  0 for bare cells
        =  1 for veg cells with a bare cell immediately left
    """

    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    arr = np.ones([ncol+ 2*edge, nrow + 2*edge])
    arr[edge:-edge, edge:-edge] = isvegc
    a = pd.DataFrame(arr) != 0

    df1 = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0).astype(int)
    df1 = np.array(df1)[edge:-edge, edge:-edge]
    df1[isvegc == 0] = 0

    df1[df1>saturate] = saturate
    
    return df1


# def func_d2lB(isvegc, edge):
#     """
#     input:
#       isvegc : [ncol x nrow] array of vegetation field
#
#     output :
#       d2lB : [ncol x nrow] array of distane to nearest left bare
#         =  0 for bare cells
#         =  1 for veg cells with a bare cell immediately left
#     """
#     d2lB = np.zeros_like(isvegc, dtype = float)
#     ncol = isvegc.shape[0]
#     nrow = isvegc.shape[1]
#
#     for k in range(nrow):
#         for j in range(ncol-1,-1,-1):
#             if isvegc[j,k] == 1:
#                 i = 1
#                 stopFlag = 0
#                 while stopFlag == 0:
#                     if isvegc[j-i,k] == 0:
#                         d2lB[j,k] = i
#                         stopFlag = 1
#                     elif j- i< 0:
#                         d2lB[j,k] = i + edge
#                         stopFlag = 1
#                     else:
#                         i = i+1
#     return d2lB
        
def func_d2lV(isvegc, edge, saturate):
    """
    input: 
      isvegc : [ncol x nrow] array of vegetation field

    output : 
      d2lV : [ncol x nrow] array, distane to nearest veg cell to the left
        =  0 for veg cells
        =  1 for bare cells with a veg cell immediately left  
    """

    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    arr = np.zeros([ncol+ 2*edge, nrow + 2*edge])
    arr[edge:-edge, edge:-edge] = 1 - isvegc
    a = pd.DataFrame(arr) != 0

    df1 = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0).astype(int)
    df1 = np.array(df1)[edge:-edge, edge:-edge]
    df1[isvegc == 1] = 0

    df1[df1>saturate] = saturate

    return df1

    # d2lV = np.zeros_like(isvegc, dtype = float)
    # ncol = isvegc.shape[0]
    # nrow = isvegc.shape[1]
    #
    # for k in range(nrow):
    #     for j in range(ncol-1,-1,-1):
    #         if isvegc[j,k] == 0:
    #             i = 1
    #             stopFlag = 0
    #             while stopFlag == 0:
    #                 if isvegc[j-i, k] == 1:
    #                     d2lV[j,k] = i
    #                     stopFlag = 1
    #                 elif j - i < 0:
    #                     d2lV[j,k] = i + edge
    #                     stopFlag = 1
    #                 else:
    #                     i = i+1



def func_d2rB(isvegc, edge, saturate):
    """
    input:
      isvegc : [ncol x nrow] array of vegetation field

    output :
      d2rB : [ncol x nrow] array;  distane to nearest bare cell to right
        =  0 for bare cells
        =  1 for veg cells with a veg cell immediately to right
    """

    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    arr = np.ones([ncol+ 2*edge, nrow + 2*edge])
    arr[edge:-edge, edge:-edge] = np.flipud(isvegc)
    a = pd.DataFrame(arr) != 0

    df1 = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0).astype(int)
    df1 = np.array(df1)[edge:-edge, edge:-edge]
    df1 = np.flipud(df1)
    df1[isvegc == 0] = 0
      
    df1[df1>saturate] = saturate
    
    return df1

# def func_d2rB(isvegc,edge):
#     """
#     input:
#       isvegc : [ncol x nrow] array of vegetation field
#
#     output :
#       d2rB : [ncol x nrow] array;  distane to nearest bare cell to right
#         =  0 for bare cells
#         =  1 for veg cells with a veg cell immediately to right
#     """
#
#
#     d2rB = np.zeros_like(isvegc, dtype = float)
#     ncol = isvegc.shape[0]
#     nrow = isvegc.shape[1]
#
#     for k in range(nrow):
#         for j in range(ncol):
#             if isvegc[j,k] == 1:
#                 i = 1
#                 stopFlag = 0
#                 while stopFlag == 0:
#                     try:
#                         if isvegc[j+i, k] == 0:
#                             d2rB[j,k] = i
#                             stopFlag = 1
#                         else:
#                             i = i+1
#                     except IndexError:
#                             d2rB[j,k] = i + edge
#                             stopFlag = 1
#     return d2rB
    
def func_d2rV(isvegc, edge, saturate):
    """
    input: 
      isvegc : [ncol x nrow] array of vegetation field

    output : 
      d2rV : [ncol x nrow] array;  distane to nearest veg cell to right
        =  0 for veg cells
        =  1 for bare cells with a veg cell immediately to right  
    """
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    arr = np.zeros([ncol+ 2*edge, nrow + 2*edge])
    arr[edge:-edge, edge:-edge] = 1- np.flipud(isvegc)
    a = pd.DataFrame(arr) != 0

    df1 = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0).astype(int)
    df1 = np.array(df1)[edge:-edge, edge:-edge]
    df1 = np.flipud(df1)
    df1[isvegc == 1] = 0

    df1[df1>saturate] = saturate

    return df1
    # d2rV = np.zeros_like(isvegc, dtype = float)
    # ncol = isvegc.shape[0]
    # nrow = isvegc.shape[1]
    #
    # for k in range(nrow):
    #     for j in range(ncol):
    #         if isvegc[j,k] == 0:
    #             i = 1
    #             stopFlag = 0
    #             while stopFlag == 0:
    #                 try:
    #                     if isvegc[j+i, k] == 1:
    #                         d2rV[j,k] = i
    #                         stopFlag = 1
    #                     else:
    #                         i = i+1
    #                 except IndexError:
    #                         d2rV[j,k] = i + edge
    #                         stopFlag = 1

                      

def get_patchL(isvegc, edge, saturate):
    """
    input : isvegc from get_source(df)  
    
    output : 
  
      patchLv:  vegetated patch length
      patchLb:  upslope interspace patch length (paired to veg patch)
      patchLc:  charcteristic length  Lv/(Lv + Lb)
      
      Ldict:  dictionary of veg patch lengths. 
         Ldict key :  downslope patch coordinate 
      Bdict:  dictionary of paired upslope interspace lengths. 
  
    usage: 
      patchLv,patchLb,patchLc,Ldict,Bdict = get_patchL(isvegc, edge)
    """
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    patchLv = np.zeros(isvegc.shape, dtype = float)  # veg patch length
    patchLb = np.zeros(isvegc.shape, dtype = float)  # upslope interspace patch length (paired to veg patch)
    
    for i in range(ncol):  # loop over across-slope direction first
        count = 0           
        for j in range(nrow):    
            if isvegc[i, j] == 1:    #  if veg patch, add 1
                if j >= (nrow -1):  # if we're at the top of the hill                  
                  patchLv[i, j-count:] = count  # record veg patch length                  
                count += 1  
                                                        
            # if [i,j] is bare and the slope cell is vegetated, record.
            # each patch starts at [i,j-count] and ends at [i,j-1]
            elif isvegc[i,j] == 0 and isvegc[i, j-1] == 1:   
                if j > 0:
                  # veg patch starts at j-count and ends at j
                  patchLv[i, j-count:j] = count
                  try:
                      # find the nearest upslope veg cell
                      Lb = np.where(isvegc[i,j:] == 1)[0][0]                               
                      patchLb[i,j-count:j] = Lb
                  except IndexError:  # bare patch extends to top of hill
                      patchLb[i,j-count:j] = nrow - j
                  count = 0 
    patchLv[patchLv > saturate] = saturate
    patchLb[patchLb > saturate] = saturate
        
    return  patchLv, patchLb


def upslope_memory(isvegc, edge, memory = 3):
    """
    
    """
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    
    dum = isvegc.copy()
    for k in range(int(nrow - memory)):
        dum[:, k] = isvegc[:, k:k+memory].mean(1)
    for k in range(1,memory):    
        dum[:, -k] = isvegc[:, -k:].mean(1)
    dum[isvegc == 0] = 0
    return dum