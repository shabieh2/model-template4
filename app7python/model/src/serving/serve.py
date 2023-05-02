#from fastapi import FastAPI
from flask import Flask, request, jsonify, render_template
#import uvicorn
import mlflow
import lightgbm as lgb
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
import logging
import sys
import socket
import tempfile
import joblib
import boto3





import numpy as np
import pickle

app = Flask(__name__)
#app.config['CORS_HEADERS'] = 'Content-Type'
#mlflow.set_tracking_uri('http://ml.mlplatform.click/mlflow')
#mlflow_run_id= "138b86e098ce4195bc4a5e65c1aba148"
#mlflow_run_id="1995425c722e432384a5fb90a8f5e0af"

mlflow.set_tracking_uri("http://18.236.226.221:5000")
logged_model = 'runs:/01e3c544c88548a68ad58b6ebe5e6cbb/models'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

#
#bucket = "mlflow-bucket-61dbad0"
#key = "model2.pkl"

'''
os.environ['BUCKETNAME']="mlflow-bucket-61dbad0"
os.environ['KEYNAME']="model2.pkl"

bucket= os.environ['BUCKETNAME']
key= os.environ['KEYNAME']
s3_client = boto3.client('s3')
os.environ['AWS_PROFILE'] = "default"
os.environ['AWS_DEFAULT_REGION'] = "us-west-2"

with tempfile.TemporaryFile() as fp:
    s3_client.download_fileobj(Fileobj=fp, Bucket=bucket, Key=key)
    fp.seek(0)
    model = joblib.load(fp)




model=mlflow.lightgbm.load_model(f'runs:/{mlflow_run_id}/model')

'''





@app.route('/')
def home():
    return render_template('index.html')
    

@app.route('/predict',methods=['POST'])
def predict():
      
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features)
    a = np.expand_dims(final_features, 0)
    y_pred=loaded_model.predict(a)

    #y_pred = np.argmax(prediction)
    
    
    output=y_pred[0][0]
    

    return render_template('index.html', prediction_text='Based on the Biometry and Lens Power, the PostOpSE is {:.2f}'.format(output))
    
    
    

@app.route('/results',methods=['POST'])
def results():

   
    data = request.get_json(force=True)
    predict_request=[[data['axial_length'],data['acd'],data['lens_thickness'],data['k_avg']]]
    my_request=np.array(predict_request)
    print(my_request)
    y_pred = model.predict(predict_request)
    #y_pred = np.argmax(prediction)
    
    output2=y_pred[0]
    prediction_text='Based on the Biometry and Lens Power, the PostOpSE is {:.2f}'.format(output2)
    return prediction_text



if __name__=="__main__":

    #serve(app, host='0.0.0.0', port='8000',threads= 8)
    app.run(debug=True)

   


   



    
