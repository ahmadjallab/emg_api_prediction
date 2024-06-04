import numpy as np

from tensorflow import keras
import os
import pickle
from static.feature_extraction import features_estimation
import pandas as pd

def pridectionModel(static_dir,data_rowEMG, extraction_features, channelName='f1', modelNumber=0, fs=2048, frame=500, step=500):
    '''
    this function predict from model list
    collect list for all model file h5 this form para model number choose from it
    :para data_rowEMG =numpy array 2d gesture row and cal vector for one  channel for each gesture
    :para extraction_feature: list for feature for model predict
    @para : modelNumber : int value for model files list

    :return = list for prediction value for each gesture **shape (gesture , each window predict )
    '''


    #print(os.listdir(static_dir))
    modelFilesList = ['2D_ANN_BinaryClassification_0.h5','2knn_model0',"2svm_model0"]
    #print("prediction function start")
    data_rowEMG = np.array(data_rowEMG)

    ##feature extraction
    gestures_features_list = []  ##each row represent label for gesture
    print(data_rowEMG.shape)
    for Gesture in data_rowEMG:
        total_feature_matrixpd_prediction, _, _, total_feature_matrix_np_prediction = features_estimation(
            signal=Gesture, channel_name=channelName, fs=fs, frame=frame, step=step)
        gestures_features_list.append(total_feature_matrixpd_prediction.loc[extraction_features].T.to_numpy().tolist())

    if modelNumber == 0:
        model_prediction = keras.models.load_model(f'{static_dir}/{modelFilesList[modelNumber]}')
    #load model
    if modelNumber == 1:
        model_prediction = pickle.load( open(f'{static_dir}/{modelFilesList[modelNumber]}', 'rb'))
    if modelNumber == 2:
        model_prediction = pickle.load( open(f'{static_dir}/{modelFilesList[modelNumber]}', 'rb'))




    #prediction section
    predicton_list = []
    for indexLabel , gesture_ in enumerate( gestures_features_list):
        prediction_value = model_prediction.predict(np.array(gesture_))
        if modelNumber == 0:
            prediction_value= np.round(np.max(prediction_value,axis=1))# for tensorflow give this vector shape [1.0] so remove [] in this way or other way by use reshape
        predicton_list.append(prediction_value.tolist())  # 0

    return predicton_list




'''
dummyData = np.random.rand(1,35)*100
gesture_name=["openHand","closeHand"]
extraction_features=['VAR', 'RMS']
prediction_list = pridectionModel(os.curdir,dummyData,extraction_features=extraction_features,modelNumber=1,fs=10, frame=5, step=5)

print (pd.DataFrame(prediction_list,index=gesture_name))

'''

