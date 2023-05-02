from fastapi import FastAPI
from flask import Flask, request, jsonify, render_template
import uvicorn
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
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
mlflow.set_tracking_uri('http://ml.mlplatform.click/mlflow')
mlflow_run_id="61c91ecf1daa47c690418b87953c3e27"


#bucket = "shs-mlflow-bucket-c8d7620"
#key = "model.sav"

bucket= os.environ['BUCKETNAME']
key= os.environ['KEYNAME']
s3_client = boto3.client('s3')
#os.environ['AWS_PROFILE'] = "default"
#os.environ['AWS_DEFAULT_REGION'] = "us-west-2"

with tempfile.TemporaryFile() as fp:
    s3_client.download_fileobj(Fileobj=fp, Bucket=bucket, Key=key)
    fp.seek(0)
    model = joblib.load(fp)

model=mlflow.lightgbm.load_model(f'runs:/{mlflow_run_id}/model')
#model=lgb.Booster(model_file='model/savedmodel.mdl')

class Size(BaseModel):

    length:float
    width:float

class PredictRequest(BaseModel):

    sepal: Size
    petal: Size
    

hostname = socket.gethostname()

version = f"{sys.version_info.major}.{sys.version_info.minor}"


flower_name_by_index = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

@app.route('/')
def home():
    return render_template('index.html')
    

@app.route('/predict',methods=['POST'])
def predict():
      
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    y_pred = np.argmax(prediction)
    
    
    
    
    
    
    
    
    
    
    output=flower_name_by_index[y_pred]

    return render_template('index.html', prediction_text='Iris type is {}'.format(output))
    #return output
    

@app.route('/results',methods=['POST'])
def results():

   
    data = request.get_json(force=True)
    predict_request=[[data['sepal_length'],data['sepal_width'],data['petal_length'],data['petal_width']]]
    my_request=np.array(predict_request)
    print(my_request)
    prediction = model.predict(predict_request)
    y_pred = np.argmax(prediction)
    
    output2=flower_name_by_index[y_pred]
    return jsonify(output2)
    
    
def main():

    app.run(debug=True)

if __name__ == "__main__":

    main()
