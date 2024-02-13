import sys
from src.ExceptionLoggerAndUtils.exception import CustomException

from src.Ppln.Ppln01dataIngestion import dataIngestionClass
from src.Ppln.Ppln02Preprocessing import preprocessingInitiatorClass
from src.Ppln.Ppln03transformationAndTraining import splittingTransformationAndTrainingInitiatorClass


try:

    obj = dataIngestionClass()
    obj.dataIngestionMethod()
    
except Exception as e:
    raise CustomException(e, sys)


try:

    obj = preprocessingInitiatorClass()
    obj.preprocessingInitiatorMethod()
    
except Exception as e:
    raise CustomException(e, sys)

try:

    obj= splittingTransformationAndTrainingInitiatorClass()
    obj.splittingTransformationAndTrainingMethond()
    
except Exception as e:
    raise CustomException(e, sys)
