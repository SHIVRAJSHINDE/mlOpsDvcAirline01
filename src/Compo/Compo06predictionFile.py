import sys
import pandas as pd
from ExceptionLoggerAndUtils.logger import App_Logger
from ExceptionLoggerAndUtils.exception import CustomException
from ExceptionLoggerAndUtils.utils import load_object

class predictionClass():
    def __init__(self):
        pass

    def predictionMethod(self,features):
        try:
            modelPath = "artifacts/model.pkl"
            preprocessorPath = "artifacts/transformation.pkl"
            model = load_object(file_path=modelPath)
            transformation = load_object(file_path=preprocessorPath)
            dataScaled = transformation.transform(features)
            pred = model.predict(dataScaled)
            return pred
        except Exception as e:
            raise CustomException(e, sys)



