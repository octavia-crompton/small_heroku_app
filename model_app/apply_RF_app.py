import sys,os
from sklearn.preprocessing import Imputer
import pickle
from glob import glob

mymods = ['ravel_fxns_app']
for mymod in mymods:
    if mymod in sys.modules:
        print mymod 
        del sys.modules[mymod]


from ravel_fxns_app import *   


def get_rvl_dirs(current_dir):
    """
    input: 
    directory with the ravel.pklz

    output: 
    paths to directories with a random forest

    """

    possible_rvl_dirs =  glob("{0}/*/".format(current_dir))
    rvl_dirs =  [rvl_dir[:-1] for rvl_dir in possible_rvl_dirs if 'ravel.pklz' in os.listdir(rvl_dir) ]
    
    
    return rvl_dirs


def get_RF_dirs(rvl_dir):
    """
    input: 
    directory with the ravel.pklz

    output: 
    paths to directories with a random forest

    """

    possible_RF_dirs =  glob("{0}/*/".format(rvl_dir))
    RF_dirs =  [RF_dir[:-1] for RF_dir in possible_RF_dirs if 'RF.sav' in os.listdir(RF_dir) ]
    
    return RF_dirs
    


def load_RF(regr_dir):
    """
    load random forest given directory
  
    tasks:
      load json file from regr_dir --> 
        bare features, veg_features, target_col  (zinflc)
  
      load RF.sav -->  regr_veg,regr_bare
    
    
    """

    regr_file = '/'.join([regr_dir,  'RF.sav' ])
    regr_veg, regr_bare, RF_summary = pickle.load(open(regr_file, 'rb'))
        
    
    bare_features = RF_summary[u'bare_features']
    veg_features = RF_summary[u'veg_features']  
    try:
      rvl_params = RF_summary[u'ravel_params']
    except:
      rvl_params = "default"
    try:
      target_col = RF_summary['target']
    except:
      target_col = 'zinflc'
      
    RF = {'bare_features' : bare_features,
          'veg_features': veg_features, 
          'target_col' :target_col,
          'regr_veg' : regr_veg,
          'regr_bare' : regr_bare,   
          'rvl_params' : rvl_params
         }
    return pd.Series(RF)
    
    
def get_scores(sim, RF):
  """
  input:
    sim: SVE simulation dict
    RF : random forest dict ()
  
  output: 
    scores, RF_zinflc
  tasks:
    call score_func  --> RF_zinflc, scores
    get infl_frac from sim.zinflc
    get RF_infl_frac from RF_zinflc
  
    
  """
  
  RF_pred, scores = score_func(sim, RF)
  
  rainD = sim.tr/60.*sim.p
  
  target_col = RF.target_col
  
  if target_col =='zinflc':
      RF_infl_frac = np.mean(RF_pred)/rainD
      infl_frac = np.mean(sim.zinflc)/rainD 
      
      scores.update ( {'RF_infl_frac': RF_infl_frac, 
                     'infl_frac' : infl_frac,
                     'd_infl_frac' : RF_infl_frac - infl_frac,
             })
 
     
  scores.update ( {'RF_pred':   np.mean(RF_pred),
                   'SVE_pred' : np.mean(sim[target_col]),
                    'd_pred' : np.mean(RF_pred) - np.mean(sim[target_col]),
            })
             
         
  return scores, RF_pred

def score_func(sim, RF):
    """
    function to reassemble map from veg and bare classifiers
    
    inputs: 
      sim, RF
  
    output:
      RF_pred : random forest predicted 
  
    tasks:
      call redo_ravel
      get ravel_veg, ravel_bare, X_veg, X_bare
      compute RF_pred by applying regr_veg and regr_bare separately
      
    """
    
    regr_veg = RF['regr_veg']
    regr_bare = RF['regr_bare']    
    bare_features = RF['bare_features']        
    veg_features = RF['veg_features']  
    features  = list(set(bare_features + veg_features))                    
    
    try:  
      target_col = RF['target_col']  
    except:
      target_col = 'zinflc'
      
    ravel, sim  = redo_ravel(sim, features, RF.rvl_params, target_col)
    ravel_veg = ravel[ravel.isvegc == 1]
    ravel_bare = ravel[ravel.isvegc == 0]
    
    X_veg, y_veg  = get_Xy(ravel_veg, veg_features, target_col)
    X_bare, y_bare = get_Xy(ravel_bare, bare_features, target_col)
    
    veg = sim['isvegc'].copy()
    RF_pred = np.zeros_like(veg, dtype = float)
    
    scores = {}
    if np.mean(veg) >0:
      RF_pred[veg == 1] = regr_veg.predict(X_veg)    
      scores.update (sub_scores(X_veg, y_veg, regr_veg, '_veg'))
    if np.mean(veg) <1:
      RF_pred[veg == 0] = regr_bare.predict(X_bare)
      scores.update (sub_scores(X_bare, y_bare, regr_bare, '_bare'))    
    
    return RF_pred, scores

def sub_scores(X, y, regr, suffix):
  """
  
  """
  predicted = regr.predict(X)
  score_dict = {}
  score_dict['r2{0}'.format(suffix)] = np_r2_score(y, predicted + np.random.rand(len(predicted))*1e-10)
  score_dict['RF{0}'.format(suffix)] = predicted.mean() 
  score_dict['SVE{0}'.format(suffix)] = y.mean()
  score_dict['d{0}'.format(suffix)] = y.mean() - predicted.mean() 
  score_dict['RMSE{0}'.format(suffix)] = np.sqrt(np.sum((y- predicted)**2)/len(predicted))
 
  return score_dict

def np_r2_score(x,y):
  
  return np.corrcoef(x, y)[0,1]**2
  
  
def unite_veg_bare(sim,RF):
    """
    function to assemble infiltration map from veg and bare classifiers
  
    analogue : get_RF_infl 
    
    used by: 
      ipython notebooks
    
    input : 
        sim : simulation dictionary 
        RF : trained RF classifier

    tasks: 
        call redo_ravel to get ptrn_ravel, updated sim 
        divide to ravel_veg, ravel_bare
        split to X_veg, y_veg ...
        apply regr_veg, regr_bare to X_veg, X_bar
    
    use:
        compare to sim.target_col      
    """
    
    regr_veg = RF['regr_veg']
    regr_bare = RF['regr_bare']    
    bare_features = RF['bare_features']        
    veg_features = RF['veg_features']  
    features  = list(set(bare_features + veg_features))                    
    
    # try:
    #   target_col = RF['target_col']
    # except:
    #   target_col = 'zinflc'
            
    ravel, sim = redo_ravel(sim, features, RF.rvl_params, None)
    
    ravel_veg = ravel[ravel.isvegc == 1]
    ravel_bare = ravel[ravel.isvegc == 0]
    
    X_veg, y_veg  = get_Xy(ravel_veg, veg_features, 'isvegc')
    X_bare, y_bare  = get_Xy(ravel_bare, bare_features, 'isvegc')

    veg = sim['isvegc'].copy()
    RF_pred = np.zeros_like(veg, dtype = float)
    RF_pred[veg == 1] = regr_veg.predict(X_veg)
    RF_pred[veg == 0] = regr_bare.predict(X_bare)

    return RF_pred
    

def redo_ravel(sim, features, rvl_params, target_col = None):
    """
    ravels veg field, updates sim with pattern_dict 
  
    analogue : get_pattern_ravel in run_pattern.py
        difference: includes zinflc
    
    input: 
      sim, veg features, bare features
    
    tasks:
      get pattern_dict from RF_patterns  (in ravel_fxns_RF.py)
      
      for all the keys in features:
        if in pattern_dict, np.ravel and add to pattern_ravel
        if in sim dict (non-array), add as list with equiv size
        skip 'local_path' key
        add 'd2divide' manually (units : grid cell)
    
    """
    isvegc = sim['isvegc'].astype(float)
    dx = sim['dx']
    
    ncol = isvegc.shape[0]
    nrow = isvegc.shape[1]
    
    xc = np.arange(0, ncol*dx, dx)  + dx/2
    yc = np.arange(0, nrow*dx, dx)  + dx/2
    xc, yc = np.meshgrid(xc, yc)
    
    xc = xc.T
    yc = yc.T
            
    pattern_dict = RF_patterns_feat(isvegc, rvl_params, features)    
    for key in pattern_dict.keys():   
      sim[key] = pattern_dict[key]   
    
    pattern_ravel = {}
    eqv_size =  int(np.size(isvegc.ravel()))
    for key in features:
        if key in pattern_dict:
            pattern_ravel[key] = np.ravel(pattern_dict[key])
        
        elif key in sim.keys() and type(sim[key]) != np.ndarray:            
            pattern_ravel[key] = np.array([sim[key]]*eqv_size)  
        
        elif key == 'local_path':
            continue
        
        elif key == 'zinflc':
            continue            
        
        elif key =='d2divide':
            pattern_ravel[key] = (nrow - yc/dx).ravel() # ( nrow - yc/sim.dx).ravel()            
        
        # else:
            # print key
    
    pattern_ravel['isvegc'] = np.ravel(isvegc)
    pattern_ravel['fV'] = np.array([np.mean(isvegc)]*eqv_size)
    
    if target_col:
        pattern_ravel[target_col] = np.ravel(sim[target_col])        
      
    pattern_ravel = impute_sim(pattern_ravel)
    pattern_ravel = pd.DataFrame(pattern_ravel)
    
    return pattern_ravel, sim

def impute_sim(sim_ravel):
  """
  update this.
  """
  cols = ['d2dB', 'd2dV',  'd2uB', 'd2uV', 'd2xB',
          'd2xV', 'd2yB', 'd2yV']
  #
  # for key in sim_ravel.keys():
  #     print key, len(sim_ravel[key])-10000

  sim_ravel = pd.DataFrame(sim_ravel)
  
  cols = [col for col in cols if col in sim_ravel.columns]
  imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
  imp = imp.fit(sim_ravel[cols])

  sim_ravel[cols] = imp.transform(sim_ravel[cols])
  
  return dict(sim_ravel)

def get_Xy(ravel, feature_cols, target_col):
    """
    ravel = ravel_bare or ravel_veg
    feature_cols = features_bare or features
    """

    X = {}
    for key in feature_cols:
        # if np.std(ravel[key])>0:
        X[key] = ravel[key] #(ravel[key]-np.mean(ravel[key]))/np.std(ravel[key]) 
        # else:
            # X[key] = 0
    y = ravel[target_col]
    X = pd.DataFrame(X)
    y = pd.Series(y)
    return X, y
    