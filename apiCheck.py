import requests 
import numpy as np
import pandas as pd
def checkAPI():
    base='https://apiemgprediction-nh76iukpta-df.a.run.app/'
    base = "http://127.0.0.1:5000/"

    #post request
    
    dummyDataPredict = np.random.rand(10000)*100

    gesture_name=["openHand","closeHand"]
    extraction_features=["VAR", "RMS"]
    channelName = ["C3"]
    modelNumber=0
    fs= 2500
    frame=500
    step= 500
    schema={"signal_emg": dummyDataPredict.tolist(),
          "extraction_features":extraction_features,
          "channelName":channelName,
          "modelNumber":modelNumber, 
          "fs":fs,
          "frame":frame,
          "step":step
          }
    response = requests.post(url=base + "api/v1/prediction-rowSignals", json=schema).json()
    
    print(response)
    #print (pd.DataFrame(response.json(),index=gesture_name))
    
    #get request
   # response = requests.get(base + "api/v1/prediction-rowSignals")
    #print(response.json())

checkAPI()