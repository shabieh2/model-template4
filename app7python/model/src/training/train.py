from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Activation, Dense, LeakyReLU
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import pandas as pd
from tensorflow import keras
import keras
import mlflow
import mlflow.lightgbm
import mlflow.keras
import boto3
import tempfile
import joblib
import os



#mlflow.set_tracking_uri('http://ml.mlplatform.click/mlflow')
mlflow.set_tracking_uri("http://18.236.226.221:5000")
mlflow.set_experiment('Lens Tensorflow Model')
mlflow.keras.autolog()

keras.backend.clear_session()

# Prepare training data
#df = pd.read_csv('./model/data/lens.csv')

os.environ['BUCKETNAME']="mlflow-bucket-61dbad0"
os.environ['FILENAME']="testData.csv"

#bucket = "mlflow-bucket-61dbad0"
#file_name = "lens2.csv"

bucket= os.environ['BUCKETNAME']
file_name= os.environ['FILENAME']


s3 = boto3.client('s3')
# 's3' is a key word. create connection to S3 using default config and all buckets within S3
obj = s3.get_object(Bucket= bucket, Key= file_name)
# get object and file (key) from bucket
data = pd.read_csv(obj['Body']) # 'Body' is a key word
#data=data.dropna()





cols_to_use =['WTW','KSteep','KFlat','AxialLength','PowerImplanted']
data_to_use = data[cols_to_use]
X = data_to_use.copy(deep = True)
y = data['PostopSE']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=103)
trainset,testset = train_test_split(data,test_size=0.2, random_state=103
)

try:
    os.mkdir('data')
except FileExistsError:
    pass

f=open("./data/source_file.txt", "a+")
f.write("s3://"+bucket+"/"+file_name)
f.close()


trainset.to_csv("./data/training_set.csv", header=True)
testset.to_csv("./data/testing_set.csv", header=True)

def baselineModel(params):
    model = tf.keras.Sequential()
    model.add(Dense(1, use_bias=False, input_shape=(params['input_shape'],)))
    model.add(LeakyReLU())
    model.add(Dense(1, use_bias=False, activation = 'linear'))
    model.add(Dense(1, activation = 'linear'))
    
    
    
    
    
    
    
    return model

'''
try:
    os.mkdir('training_1')
except FileExistsError:
    pass

path_checkpoint = "./training_1/cp.ckpt"
directory_checkpoint = os.path.dirname(path_checkpoint)

callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                             save_weights_only=True,
                                             verbose=1)

'''
params= {'input_shape': data_to_use.shape[1]}
model = baselineModel(params)


optimizer=tf.keras.optimizers.SGD(clipvalue=1)

model.compile(optimizer, loss='mean_squared_error')

results=model.fit(X_train, y_train, epochs=200, batch_size=64,validation_data=(X_test,y_test))






def main():
  with mlflow.start_run(run_name = 'lens_tf_model_0413') as run:
  
    mlflow.keras.log_model(model, "models")
  
    
  
    
    
    

#print("BASE DIR",BASE_DIR_NEW)
    model.save("model.keras")

    # Evaluate model
    y_pred = model.predict(X_test)
    #y_pred = y_proba.argmax(axis=1)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rsquared = r2_score(y_test, y_pred)
    #rec = recall_score(y_test, y_pred,average='weighted')
    
    

    # Log metrics
    mlflow.log_metrics({
      "mse": mse,
      "mae": mae,
            "rsquared": rsquared,
                  
    })
    
    mlflow.log_artifact("data",artifact_path="training_data_path")
    

    print("Run ID:", run.info.run_id)
    keras.backend.clear_session()
  

if __name__ == "__main__":
    main()
