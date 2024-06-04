# Description: This is the main file that will run the Flask server. It will be responsible for handling the incoming requests and sending the responses back to the client.
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from static.prediction import pridectionModel
from flask_cors import CORS
import os
app = Flask(__name__)
api = Api(app)
CORS(app)  # This will enable CORS for all routes and methods

def dataValidation_forPredictionRowSignals_endpoint (json_schema):
    required_fields = ["signal_emg", "extraction_features", "channelName", "modelNumber", "fs", "frame", "step"]
    #print ("Validating data")
    for field in required_fields:
        if field not in json_schema:
            return False
    for field in json_schema:
        if not isinstance(json_schema[field], (int, list)):  # adjust this line based on the expected data types
            return False
        if not json_schema[field] :
            if field != "modelNumber":
                return False
    return True


class PredictionRowSignals(Resource):
    def post(self):
        #print(app.static_folder)
        #print (os.listdir(app.static_folder))
        data = request.json
        print(data)
        gestures = data["signal_emg"]
        if len(gestures) > 2 :
            return {'error':'error shape for gesture matrix '}
        if not gestures:
            return {'error': "signals_ emg is empty"}
        for gesture in gestures:
            if len(gesture)<data["fs"] or len(gesture)<data["frame"] or len(gesture)<data["step"] :
                return {'error': "signals_ emg  less than fs or frame or step"}

        if dataValidation_forPredictionRowSignals_endpoint(data):
            #print("full fields  \n ")
           # print(data)
            return pridectionModel(app.static_folder,gestures, data["extraction_features"], data["channelName"], data["modelNumber"], data["fs"], data["frame"], data["step"])
        else:
            #print("miss fields \n ")
            return  pridectionModel(app.static_folder,data_rowEMG=data['signal_emg'])

    
    def get(self):
        return {"message": "API schema", "schema": {
            "method": "POST",
            
            "data": {
                "signal_emg": "2D array of EMG signals with shape (gustures, samples) ",
                "extraction_features":"list of features to extract from the signals",
                "channelName" : "list string",
                 "modelNumber": "int for the model number" , 
                 "fs": "int for the sampling frequency", 
                 "frame": "int for the frame size",
                 "step": "int for the step size",

            },
           "notes": {
                "model_map_label": {
                    '2D_ANN_BinaryClassification_0.h5': 0,
                    '2knn_model0': 1,
                    '2svm_model0': 2
                },
               "data" : "row emg more than fs and frame and step size "
            }

        }}

class Home(Resource):
    def get(self):
        return {"message": "Welcome to the EMG API" , "endpoints": {"/api/v1/prediction-rowSignals": "POST: Send EMG signals to get predictions, GET: Get API schema"}}



api.add_resource(Home, "/")
api.add_resource(PredictionRowSignals, "/api/v1/prediction-rowSignals")




if __name__ == "__main__":
    app.run(debug=True)