from fastapi import FastAPI
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
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app=FastAPI()
mlflow.set_tracking_uri('http://ml.mlplatform.click/mlflow')



#Load model



mlflow_run_id="ce01686f43fe43dd87416f356061a775"

#model=mlflow.lightgbm.load_model("runs:/80d7a42c307240ebbeb8bb0d3c973834/model")
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



templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(request: PredictRequest):
    df = pd.DataFrame(columns=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'],
                      data=[[request.sepal.length, request.sepal.width, request.petal.length, request.petal.width]])
                      
    y_pred = np.argmax(model.predict(df))
    
    return {"flower": flower_name_by_index[y_pred]}
    

                      
@app.get("/predict")
async def read_root():
 return {
        "name": "my-app",
        "host": hostname,
        "version": f"Hello world! From FastAPI and Shabieh running on Uvicorn. Using Python {version}"
    }

@app.post("/predict2")
def predict(request: PredictRequest):
    df = pd.DataFrame(columns=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'],
                      data=[[request.sepal.length, request.sepal.width, request.petal.length, request.petal.width]])

    y_pred = np.argmax(model.predict(df))
    #return {"flower": flower_name_by_index[y_pred]}
    return {"flower": int(y_pred)*2}


def main():
    
    uvicorn.run(app)
    logging.info('Main app sequence begun')
    logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    logging.info('App finished')

if __name__ == "__main__":

    logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    main()
    logging.info('Main app sequence begun')
    logging.info('App finished')




