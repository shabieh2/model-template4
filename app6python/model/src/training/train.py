from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import boto3
import tempfile
import joblib
import os

import pandas as pd


bucket = "mlflow-bucket-61dbad0"
file_name = "iris.csv"

#bucket= os.environ['BUCKETNAME']
#key= os.environ['KEYNAME']
#file_name= os.environ['FILENAME']
#os.environ['AWS_PROFILE'] = "default"
#os.environ['AWS_DEFAULT_REGION'] = "us-west-2"


s3 = boto3.client('s3')
# 's3' is a key word. create connection to S3 using default config and all buckets within S3
obj = s3.get_object(Bucket= bucket, Key= file_name)
# get object and file (key) from bucket
df = pd.read_csv(obj['Body']) # 'Body' is a key word



# Prepare training data
#df = pd.read_csv('model/data/iris.csv')

flower_names = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}


X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = df['variety'].map(flower_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

train_data = lgb.Dataset(X_train, label=y_train)

def main():


    
    # Train model
    params = {
      "objective": "multiclass",
      "num_class": 3,
      "learning_rate": 0.2,
      "metric": "multi_logloss",
      "feature_fraction": 0.8,
      "bagging_fraction": 0.9,
      "seed": 40,
    }

    model = lgb.train(params, train_data, valid_sets=[train_data])

    # Evaluate model
    y_proba = model.predict(X_test)
    y_pred = y_proba.argmax(axis=1)

    loss = log_loss(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    # Log metrics
    
    print("loss: ",loss,"acc: ",acc)
    
    #model.save_model('model/savedmodel.mdl')
    #s3.put_object(Body=model, Bucket=bucket,Key='savedmodel.mdl')
    
    

    s33 = boto3.resource('s3')

# you can dump it in .sav or .pkl format
    location = '/' # THIS is the change to make the code work
    model_filename = 'model2.pkl'  # use any extension you want (.pkl or .sav)
    OutputFile = model_filename

# WRITE
    with tempfile.TemporaryFile() as fp:
        joblib.dump(model, fp)
        fp.seek(0)
        # use bucket_name and OutputFile - s3 location path in string format.
        s33.Bucket(bucket).put_object(Key= OutputFile, Body=fp.read())
    print(f"----------------------------------------------------------MODEL has been stored in {bucket} in AWS S3 as {model_filename}----------------------------------------")

    
    
    
    
    

 
  

if __name__ == "__main__":
    main()
