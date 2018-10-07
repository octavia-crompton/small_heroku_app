import os
from flask import Flask, flash, request, redirect, url_for
from flask import render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
import os
import json
import gzip, pickle
from datetime import datetime
import shutil
import numpy as np


from forms import Landscape, Upload, Storm, ImageUpdate, reset_features
from compute import run_RF, png_plot, binarize


app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'dat', 'pklz'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST']) 

def index():  
  """
  App homepage (and only page)
    
  """
  vegform = Upload()  
  imageform = ImageUpdate()    
  stormform = Storm()   
  form = Landscape() 
  
  # Set image defaults to zero so template knows what's going on
  result_im = None
  result_bin = None  
  result_RF = None
  result_vmax = None

  feature_dir = '/'.join([os.getcwd(), 'feature' + request.remote_addr])
  feature_path = '/'.join([feature_dir, 'features.json'])
  os.mkdir(feature_dir) if os.path.isdir(feature_dir) == False  else True
  if 'features.json' in os.listdir(feature_dir):
    with open(feature_path, 'r') as fp:
      features = json.load(fp)
     
  else:
    features = reset_features()

  p_choices = {'0.5' : ['2.5', '5.0', '7.5', '10'],
               '1' : ['1.7','3.4','5.1','6.8'],
               '3' : ['0.8','1.6','2.4','3.2'],
               '6' :['0.5','1.0','1.5','2.0']
                 }


  if request.method == 'POST' and  form.reset.data == True :
         flash('Reset!')
         features = reset_features()         
         with open(feature_path, 'w') as fp:
             json.dump(features, fp, indent = 2)      
  
  if 'tr' in features.keys():
      form.p.choices = [(str(p),str(p)) for p in p_choices[features['tr']]]   
  
  if vegform.validate_on_submit():
     
     if vegform.delete.data == True:
       flash('File deleted ')
       
       #os.remove(os.path.join( UPLOAD_FOLDER, features['filename'] ))
       if 'filename' in features:
           features.pop('filename')
       if 'filepath' in features:
           features.pop('filepath')                
       result_bin = None
       result_im = None 
       
       features = reset_features()      
      
     if vegform.submit.data == True:
       
       flash('File upload ')
       f = vegform.veg.data
       filename = secure_filename(f.filename)
       filepath = os.path.join( UPLOAD_FOLDER, filename)
       f.save(filepath)
     
       features['filename'] = filename     
       features['filepath'] = filepath         

  #### image handling   
  if imageform.submit.data and imageform.validate_on_submit():
     
     flash('Image update')
     keys = [key for key in request.form.keys() if key not in ['csrf_token', 'ascii']]
     #for key in keys:       
     if 'grid' in keys:
         features['grid'] = request.form['grid']
     if 'threshold' in keys:
         features['threshold'] = request.form['threshold']
  
     if request.form['submit'] == 'rotate-right':
         features['rotate'] = np.mod(features['rotate'] - 90., 360)
         print "rotate right"

     if request.form['submit'] == 'rotate-left':
         features['rotate'] = np.mod(features['rotate'] + 90., 360)
         print "rotate left"
           
  #### precip handling  
  if stormform.submit.data and stormform.validate_on_submit():
      
      flash('Rain duration added ')     
    
      features['tr'] = (request.form['tr'])
      if 'p' in features.keys():
          features['rainD']  = float(features['tr'])*float(features['p'])
      
      
      form.p.choices = [(str(p),str(p)) for p in  p_choices[features['tr']]] 
      keys =     [key for key in request.form.keys() if key not in ['csrf_token', 'ascii']]
          

  #### update landscape 
  if form.update.data and form.validate_on_submit():
     
     flash('Landscape featuress added')
     
     keys = [key for key in request.form.keys() if key not in ['csrf_token', 'ascii']]
     
     if 'tr' not in features.keys():
         features['tr'] = '0.5'
             
     features['slope'] = request.form['slope']
     features['p'] = (request.form['p'])
     features['KsV'] = (request.form['KsV'])     
     features['rainD']  = float(features['tr'])*float(features['p'])
   
  if 'filepath' in features.keys():
    
    result_im = png_plot(features['filepath'])
    
    threshold = features['threshold'] if 'threshold' in features.keys() else 0.5 
    result_bin, bw = binarize(features['filepath'], features = features)
    features['Lx'] = bw.shape[0]
    features['Ly'] = bw.shape[1]    
    features['fV'] = np.round(np.mean(bw), 2)
    
  if form.submit.data and form.validate_on_submit():
     flash('Running the random forest model!')
     
     keys = [key for key in request.form.keys() if key not in ['csrf_token', 'ascii']]
     
     if 'tr' not in features.keys():
         features['tr'] = '0.5'
         
     features['slope'] = request.form['slope']
     features['p'] = (request.form['p'])
     features['KsV'] = (request.form['KsV'])     
     features['rainD']  = float(features['tr'])*float(features['p'])
     
     result_RF, zinflc = run_RF(features['filepath'], features = features, target_col = 'zinflc')
     result_vmax, vmax = run_RF(features['filepath'], features = features, target_col = 'vmax')     
     
     features['inflDveg'] = np.round(np.mean(zinflc[bw == 1]),2)
     features['inflD'] = np.round(np.mean(zinflc),2)
     features['vmax'] = np.round(np.mean(vmax),2)
     features['vmaxmax'] = np.round(np.percentile(vmax, 95),2)     

  with open(feature_dir + '/features.json', 'w') as fp:
      json.dump(features, fp, indent = 2)      

  keys = [key for key in features.keys() if key not in ['filepath', 'filename', 'tr']]

  for key in ['grid', 'threshold']:
    if key in features.keys():
        imageform[key].default = float(features[key] )          
    
  imageform.process()

  for key in ['tr']:
    if key in features.keys():
        stormform[key].default = features[key]
        
  stormform.process()  
    
  for key in [ 'p', 'slope', 'KsV']:
    if key in features.keys():
        form[key].default = features[key]

  form.process()
  
  return render_template('home.html', form = form, \
              vegform = vegform, 
              imageform = imageform, 
              stormform = stormform, 
              features = features, 
              result_bin = result_bin,
              result_im = result_im,
              result_RF = result_RF,
              result_vmax = result_vmax
              )

if __name__ == '__main__':
  app.secret_key = 'super_secret_key'

  app.run(debug=False, use_reloader=True, threaded = True)
