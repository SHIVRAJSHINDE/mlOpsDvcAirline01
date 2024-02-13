
from src.Compo.Compo03Transformation import dataSplittingTransformationClass
from src.Compo.Compo04modelTrainingFile import modelTrainingClass

from src.ExceptionLoggerAndUtils.exception import CustomException
from src.ExceptionLoggerAndUtils.logger import App_Logger
from src.ExceptionLoggerAndUtils.utils import save_object



import os
import sys


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

            models, parameters = self.modelTrainingObj.modelsToTrainAndParameters()

            model_report: dict = self.modelTrainingObj.evaluate_Regression_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,models=models, param=parameters)

            self.modelTrainingObj.modelTraingMethod(model_report,models)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    splitTransAndTrainObj =  splittingTransformationAndTrainingInitiatorClass()
    splitTransAndTrainObj.splittingTransformationAndTrainingMethond()


