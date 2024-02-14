from src.Compo.Compo03Transformation import dataSplittingTransformationClass
from src.Compo.Compo05modelTrainingMlflowFile import modelTrainingClass
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from src.ExceptionLoggerAndUtils.exception import CustomException
from src.ExceptionLoggerAndUtils.logger import App_Logger
from src.ExceptionLoggerAndUtils.utils import save_object
import warnings
import pandas as pd
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import numpy as np
import os
import sys
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)


class splittingTransformationAndTrainingInitiatorClass():
    def __init__(self):
        self.splittingTransformObj = dataSplittingTransformationClass()
        self.modelTrainingObj = modelTrainingClass()
        self.File_Path = "artifacts/cleanedData/cleanedDataFile.csv"
        self.transformationFilePath = os.path.join('artifacts', "transformation.pkl")


    def splittingTransformationAndTrainingMethond(self):
        try:

            X_train, X_test, y_train, y_test = self.splittingTransformObj.dataReadingAndSplitting(self.File_Path)
            transformationOfData =self.splittingTransformObj.dataTransformation()
            print(X_train.T)

            X_train = transformationOfData.fit_transform(X_train)
            X_test = transformationOfData.transform(X_test)

            save_object(
                file_path= self.transformationFilePath,
                obj = transformationOfData
            )

            models, parameters = self.modelTrainingObj.mlFlowParametersAndModelsForTraining()
            
            warnings.filterwarnings("ignore")
            np.random.seed(40)
            alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
            l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

            
            with mlflow.start_run():
                lr = RandomForestRegressor(random_state=42)
                 
                lr.fit(X_train, y_train)

                predicted_qualities = lr.predict(X_test)
                
                (rmse, mae, r2) = self.modelTrainingObj.eval_metrics(y_test, predicted_qualities)

                print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
                print("  RMSE: %s" % rmse)
                print("  MAE: %s" % mae)
                print("  R2: %s" % r2)

                mlflow.log_param("alpha", alpha)
                mlflow.log_param("l1_ratio", l1_ratio)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)
                
                remote_server_uri = "https://dagshub.com/SHIVRAJSHINDE/mlOpsDvcAirline01.mlflow"
                mlflow.set_tracking_uri(remote_server_uri)


                # remote_server_uri = "http://ec2-16-171-1-148.eu-north-1.compute.amazonaws.com:5000/"
                # mlflow.set_tracking_uri(remote_server_uri)

                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                # Model registry does not work with file store
                if tracking_url_type_store != "file":
                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
                else:
                    mlflow.sklearn.log_model(lr, "model")
            
            

            '''model_report: dict = self.modelTrainingObj.evaluate_Regression_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,models=models, param=parameters)

            self.modelTrainingObj.modelTraingMethod(model_report,models)'''

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    splitTransAndTrainObj =  splittingTransformationAndTrainingInitiatorClass()
    splitTransAndTrainObj.splittingTransformationAndTrainingMethond()

