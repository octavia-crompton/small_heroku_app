import numpy as np
import pandas as pd
import scipy as sp
import scipy.ndimage
from scipy import ndimage
np.seterr(divide='ignore', invalid='ignore')

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
    try:
        gsigma = rvl_params['gsigma']
    except:
        gsigma = [int(rvl_params['sigma'])]
      
    if type(gsigma) == int:
        gsigma = [gsigma]
        

    d2uB = func_d2uB(isvegc, edge ,saturate)
    d2dB = func_d2dB(isvegc, edge ,saturate)     

    d2yB = np.minimum(d2uB, d2dB)
    d2yB[d2yB > saturate] = saturate
    
    d2lB = func_d2lB(isvegc, edge, saturate)
    d2rB = func_d2rB(isvegc, edge, saturate)    
    
    d2xB = np.minimum(d2lB, d2rB)    
    d2xB[d2xB > saturate] = saturate
                  
    d2uV = func_d2uV(isvegc, edge, saturate)
    d2dV = func_d2dV(isvegc, edge, saturate)
    d2yV = np.minimum(d2uV, d2dV)

    d2lV = func_d2lV(isvegc, edge, saturate)
    d2rV = func_d2rV(isvegc, edge, saturate)   
    d2xV = np.minimum(d2lV, d2rV)
                  
    patchLv,patchLb = get_patchL(isvegc, saturate) 
    bareL, bareLv = get_bareL(isvegc, saturate) 
    
    upslopeLs = np.hstack((np.arange(3,6) ,np.arange(6, 20, 4),  np.arange(22, 60, 8)))

    
    pattern_dict = {'isvegc' : isvegc,                                                
                    'd2uB' : d2uB, 
                    'd2dB' : d2dB, 
                    'd2xB' : d2xB,
                    'd2uV' : d2uV, 
                    'd2dV' : d2dV, 
                    'd2xV' : d2xV,                    
                    'patchLv' : patchLv, 
                    'patchLb' : patchLb,
                    'bareL' : bareL, 
                    'bareLv' : bareLv,                                        
                  }
    
    for L in upslopeLs:         
        for gs in gsigma: 
            upslopeL =  upslope_memory(isvegc,  min(nrow, int(L)))
            pattern_dict['upslope{0}'.format(L)] = upslopeL.copy()
          
            pattern_dict['upslope{0}'.format(L) + 'b_s{0}'.format(gs)] =   smoothB(upslopeL, isvegc, gs)        
            pattern_dict['upslope{0}'.format(L) + '_s{0}'.format(gs)] =   smoothV(upslopeL, isvegc, gs)        

    for gs in gsigma:           
        for key in ['d2uV','d2dV','d2xV','bareLv','bareL',  ]:
            pattern_dict[key + '_s{0}'.format(gs)] =   smoothB(pattern_dict[key], isvegc, gs)        
        

        for key in ['d2uB', 'd2dB', 'd2xB','patchLb', 'patchLv']:
            pattern_dict[key + '_s{0}'.format(gs)] =   smoothV(pattern_dict[key], isvegc, gs)        
                

    return pattern_dict                

def RF_patterns_feat(isveg, rvl_params, features):  
       
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
    try:
        gsigma = rvl_params['gsigma']
    except:
        gsigma = [int(rvl_params['sigma'])]
      
    if type(gsigma) == int:
        gsigma = [gsigma]
        

    d2uB = func_d2uB(isvegc, edge ,saturate)
    d2dB = func_d2dB(isvegc, edge ,saturate)     

    d2yB = np.minimum(d2uB, d2dB)
    d2yB[d2yB > saturate] = saturate
    
    d2lB = func_d2lB(isvegc, edge, saturate)
    d2rB = func_d2rB(isvegc, edge, saturate)    
    
    d2xB = np.minimum(d2lB, d2rB)    
    d2xB[d2xB > saturate] = saturate
                  
    d2uV = func_d2uV(isvegc, edge, saturate)
    d2dV = func_d2dV(isvegc, edge, saturate)
    d2yV = np.minimum(d2uV, d2dV)

    d2lV = func_d2lV(isvegc, edge, saturate)
    d2rV = func_d2rV(isvegc, edge, saturate)   
    d2xV = np.minimum(d2lV, d2rV)
                  
    patchLv,patchLb = get_patchL(isvegc, saturate) 
    bareL, bareLv = get_bareL(isvegc, saturate) 
    
    upslopeLs = []
    for feat in features:
        if 'upslope' in feat:
            L = feat.replace('upslope', '').split('_')[0].replace('b', '')
            upslopeLs.append(L)

    upslopeLs =  np.unique(upslopeLs).astype(int)
    
    
    pattern_dict = {'isvegc' : isvegc,                                                
                    'd2uB' : d2uB, 
                    'd2dB' : d2dB, 
                    'd2xB' : d2xB,
                    'd2uV' : d2uV, 
                    'd2dV' : d2dV, 
                    'd2xV' : d2xV,                    
                    'patchLv' : patchLv, 
                    'patchLb' : patchLb,
                    'bareL' : bareL, 
                    'bareLv' : bareLv,                                        
                  }
    
    for L in upslopeLs: 
        for gs in gsigma:  
            upslopeL =  upslope_memory(isvegc,  min(nrow, L))

            pattern_dict['upslope{0}'.format(L)] = upslopeL.copy()
        
            pattern_dict['upslope{0}'.format(L) + 'b_s{0}'.format(gs)] =   smoothB(upslopeL, isvegc, gs)        
            pattern_dict['upslope{0}'.format(L) + '_s{0}'.format(gs)] =   smoothV(upslopeL, isvegc, gs)        

    for gs in gsigma:           
        for key in ['d2uV','d2dV','d2xV','bareLv','bareL'  ]:
            pattern_dict[key + '_s{0}'.format(gs)] =   smoothB(pattern_dict[key], isvegc, gs)        
        

        for key in ['d2uB', 'd2dB', 'd2xB','patchLb', 'patchLv']:
            pattern_dict[key + '_s{0}'.format(gs)] =   smoothV(pattern_dict[key], isvegc, gs)        
                

    return pattern_dict    
    

def RF_patterns_dep(isveg, rvl_params):        
       
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
    try:
        gsigma = rvl_params['gsigma']
    except:
        gsigma = [int(rvl_params['sigma'])]
      
    if type(gsigma) == int:
        gsigma = [gsigma]
        
    d2wB = func_d2wB(isvegc, saturate, weight)      
    d2B = func_d2B(isvegc, saturate)          

    d2wV = func_d2wV(isvegc, saturate, weight)      
    d2V = func_d2V(isvegc, saturate)          

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
          
    patchLv,patchLb = get_patchL(isvegc, saturate) 
    bareL, bareLv = get_bareL(isvegc, saturate) 
    #upslope_sum = np.flip(np.cumsum(np.flip(isvegc, 1), 1), 1)
    upslope10 = upslope_memory(isvegc,  min(nrow, 10))
    upslope5 = upslope_memory(isvegc, min(nrow, 5))    
    upslope3 = upslope_memory(isvegc, min(nrow, 3))    
    
    
    pattern_dict = {'isvegc' : isvegc,
                    'd2wB' : d2wB, 
                    'd2B' : d2B,    
                    'd2wV' : d2wV, 
                    'd2V' : d2V,                                         
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
                    'bareL' : bareL, 
                    'bareLv' : bareLv,                                        
                    'upslope10' : upslope10, 
                    'upslope5' : upslope5,      
                    'upslope3' : upslope3       
                  }
    
    #.bare   
    for gs in gsigma:           
        for key in ['d2uV','d2dV','d2xV','d2yV', 'bareLv','bareL', 'upslope3', 'upslope5', 'upslope10' ]:
            pattern_dict[key + '_s{0}'.format(gs)] =   smoothB(pattern_dict[key], isvegc, gs)        
    
        for key in ['d2uB', 'd2dB', 'd2xB','d2yB','patchLb', 'patchLv', 'upslope3', 'upslope5','upslope10']:
            pattern_dict[key + '_s{0}'.format(gs)] =   smoothV(pattern_dict[key], isvegc, gs)        
            
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
    """
    smooths over V
    """
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


def func_d2wV(isvegc, saturate, weight):
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
        d = 1-isvegc.copy()
        d[:,:i+1] = 1
        res[:,i] = ndimage.distance_transform_edt(d, sampling = (weight, 1))[:, i]   
    res[isvegc ==1] = 0
    
    res[res>saturate] = saturate   
    return res
    

def func_d2B(isvegc, saturate):
    """
    Distane to  nearest upslope bare cell
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
    res[res>saturate] =  saturate    
    
    return res
    
    
def func_d2V(isvegc, saturate):
    """
    Distane to  nearest upslope bare cell
    =  0 for bare ground
    =  1 for veg cells with a neighboring bare cell upslope
    >  1 for veg cells with bare cells further upslope
    """
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]

    res =  isvegc.copy()
    
    for i in range(nrow):
        d = 1-isvegc.copy()
        d[:,:i+1] = 1
        res[:,i] = ndimage.distance_transform_edt(d, sampling = (1, 1))[:, i]   
    
    res[isvegc ==1] = 0
    res[res>saturate] =  saturate    
    
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
      

def func_d2uV(isvegc, edge, saturate):
    """
    Distane to nearest upslope veg cell
    =  0 for veg cells
    =  1 for bare cells with a neighboring veg cell upslope
    >  1 for bare cells with a  veg cell further upslope
    """
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    arr = np.ones([ncol+ 2*edge, nrow + 2*edge]).T
    arr[edge:-edge, edge:-edge] = 1 - np.flipud(isvegc.T)
    a = pd.DataFrame(arr) != 0

    df1 = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0).astype(int)
    df1 = np.array(df1)[edge:-edge, edge:-edge]
    df1 = np.flipud(df1).T
    df1[isvegc == 1] = 0

    df1[df1>saturate] = saturate  
    return df1
    

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
    arr = np.ones([ncol+ 2*edge, nrow + 2*edge]).T
    arr[edge:-edge, edge:-edge] = 1 - isvegc.T
    a = pd.DataFrame(arr) != 0

    df1 = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0).astype(int)
    df1 = np.array(df1)[edge:-edge, edge:-edge]
    df1 =  df1.T
    df1[isvegc == 1] = 0
   
    df1[df1>saturate] = saturate
    
    return df1



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
    arr = np.ones([ncol+ 2*edge, nrow + 2*edge])
    arr[edge:-edge, edge:-edge] = 1 - isvegc
    a = pd.DataFrame(arr) != 0

    df1 = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0).astype(int)
    df1 = np.array(df1)[edge:-edge, edge:-edge]
    df1[isvegc == 1] = 0

    df1[df1>saturate] = saturate

    return df1


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
    arr = np.ones([ncol+ 2*edge, nrow + 2*edge])
    arr[edge:-edge, edge:-edge] = 1- np.flipud(isvegc)
    a = pd.DataFrame(arr) != 0

    df1 = a.cumsum()-a.cumsum().where(~a).ffill().fillna(0).astype(int)
    df1 = np.array(df1)[edge:-edge, edge:-edge]
    df1 = np.flipud(df1)
    df1[isvegc == 1] = 0

    df1[df1>saturate] = saturate

    return df1
 

def get_patchL(isvegc, saturate):
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
      patchLv,patchLb,patchLc,Ldict,Bdict = get_patchL(isvegc)
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


def get_bareL(isvegc, saturate):
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
      patchLv,patchLb,patchLc,Ldict,Bdict = get_patchL(isvegc)
    """
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    bareLv = np.zeros(isvegc.shape, dtype = float)  # veg patch length
    bareL = np.zeros(isvegc.shape, dtype = float)  # upslope interspace patch length (paired to veg patch)
    
    for i in range(ncol):  # loop over across-slope direction first
        count = 0           
        for j in range(nrow):    
            if isvegc[i, j] == 0:    #  if bare, add 1
                if j >= (nrow -1):  # if we're at the top of the hill                  
                  bareL[i, j-count:] = count  # record bare length                  
                count += 1  
                                                        
            # if [i,j] is veg and the slope cell is bare, record.
            # each patch starts at [i,j-count] and ends at [i,j-1]
            elif isvegc[i,j] == 1 and isvegc[i, j-1] == 0:   
                if j > 0:
                  # veg patch starts at j-count and ends at j
                  bareL[i, j-count:j] = count
                  try:
                      # find the nearest upslope bare cell
                      Lb = np.where(isvegc[i,j:] == 0)[0][0]                               
                      bareLv[i,j-count:j] = Lb
                  except IndexError:  # bare patch extends to top of hill
                      bareLv[i,j-count:j] = nrow - j
                  count = 0 
    bareLv[bareLv > saturate] = saturate
    bareL[bareL > saturate] = saturate
        
    return  bareL, bareLv



def upslope_memory(isvegc,  memory = 3):
    """
    
    """
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    
    dum = isvegc.copy()
    memory = int(memory)
    for k in range(int(nrow - memory)):
        dum[:, k] = isvegc[:, k:k+memory].sum(1)
    for k in range(1,memory+1):    
        dum[:, -k] = isvegc[:, -k:].sum(1)
    # dum[isvegc == 0] = 0
    return dum