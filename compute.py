import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os
from skimage import color
import random
from scipy import misc
import imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable


def png_plot(filepath, resolution=200):
    """Return filename of plot of the damped_vibration function."""
  
    fig= plt.figure()
    ax = fig.add_subplot(111)                       
    plt.axis('off')
    
    #image = misc.face(gray=True)
    image = imageio.imread(filepath)
    image = color.rgb2gray(image)
    image = np.fliplr(image.T)
    im = ax.pcolormesh(image.T, cmap = 'Greens')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)    
      
    # Make Matplotlib write to BytesIO file object and grab
    # return the object's string
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png', transparent=True, bbox_inches='tight', pad_inches=0)

    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())

    return figdata_png
    
def binarize(filepath, features):
    
    """Return filename of plot of the damped_vibration function."""
    
    threshold = features['threshold']
    rotate = features['rotate']
    scale = float(features['grid'])
    
    image = imageio.imread(filepath)
    image = color.rgb2gray(image)
    image = np.fliplr(image.T)
        
    if scale != 1:
        image = regrid(image, scale)
        
    bw = 1.0*(image > float(threshold))
    
    if rotate == 90:
            bw =np.fliplr(bw).T           
    elif rotate == 180:
            bw = np.fliplr(bw)           
    elif rotate == 270:
            bw =bw.T
        
    fig= plt.figure()
    ax = fig.add_subplot(111)
    plt.axis('off')
    im = ax.pcolormesh(bw.T, cmap = 'Greens')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)       
 
    # Make Matplotlib write to BytesIO file object and grab return the object's string
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    figfile.seek(0)  # rewind to beginning of file
    
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())

    return figdata_png, bw
 
def regrid(im, scale): 
     
     from scipy.interpolate import RegularGridInterpolator
     y = np.arange(0, im.shape[0])
     x = np.arange(0, im.shape[1])
     interpolating_function = RegularGridInterpolator((y, x), im)
     
     xv = np.arange(0, im.shape[0], scale)
     yv = np.arange(0, im.shape[1], scale)

     xv, yv = np.meshgrid(xv,yv)
     
    
     # xv = xv.T
     # yv = yv.T
     
     im2 = interpolating_function((xv, yv)).T   
        
     return im2
     
     
def run_RF(filepath, features, target_col = 'zinflc'):
    
    """Return filename of plot of the damped_vibration function."""
    RF_var = {}
    RF_var['So'] = float(features['slope'])/100.
    RF_var['Ks'] = features['KsV']
    RF_var['tr'] = int(float(features['tr'])*60)
    
    RF_str =   ','.join([var  + '-' + str(RF_var[var]) for var in ['tr', 'So', 'Ks'  ] ])

    searchdir = '/'.join(['RFs', target_col, RF_str ])

    if os.path.isdir(searchdir):
        print searchdir, 'success'
        RF_dir = searchdir
    else:        
        RF_dir = 'RFs/{0}/tr-30,So-0.1,Ks-2.5'.format(target_col)
        print RF_dir, 'default'
    
    figdata_png, bw = binarize(filepath, features)
    
    RF_pred = wrap_RF(RF_dir, bw, features)
    
    fig= plt.figure()
    ax = fig.add_subplot(111)
    plt.axis('off')

    if target_col == 'zinflc':
        cmap = 'Blues'
        label = 'cm'
    elif target_col == 'vmax':
        cmap = 'YlGnBu'
        label = 'cm/s'
                
    im = ax.pcolormesh(RF_pred.T, cmap = cmap  )    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label = label)
    
    # Make Matplotlib write to BytesIO file object and grab return the object's string
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
    figfile.seek(0)  # rewind to beginning of file
    
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())

    return figdata_png, RF_pred

  
def wrap_RF(RF_dir, isvegc, features ):
    
    import sys
    import pandas as pd
    
    sys.path.append('model_app')

    mymods = [ 'apply_RF_app', 'ravel_fxns_app']
    for mymod in mymods:
        if mymod in sys.modules: 
            del sys.modules[mymod]
            
    from apply_RF_app import load_RF, unite_veg_bare 

    
    RF = load_RF(RF_dir)
    
    dx = float(features['grid'])
    
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    xc = np.arange(0, ncol*dx, dx)  + dx/2
    yc = np.arange(0, nrow*dx, dx)  + dx/2
    xc, yc = np.meshgrid(xc, yc)
    
    xc = xc.T
    yc = yc.T
    
    So = float(features['slope'])/100.
    Ks = features['KsV']
    tr = float(features['tr'])*60
    p =  float(features['p'])
        
    sim = pd.Series({'isvegc' : isvegc,
           'ncol' : ncol,
           'nrow' : nrow,
           'dx' : dx,
           'yc' : yc.T,
           'xc' : xc.T,
           'p' : p,
           'sigma' : 3
            
           })
           
    dum = unite_veg_bare(sim, RF)
    
    return dum