from src.ExceptionLoggerAndUtils.exception import CustomException
from src.ExceptionLoggerAndUtils.logger import App_Logger

from src.Compo.Compo01dataIngestion import dataReadingAndCleaningClass


import os
import sys
import pandas as pd

class dataIngestionClass():
    def __init__(self):

        self.cwd=os.getcwd()
        self.log_writer = App_Logger()
        self.preprocessingObj = dataReadingAndCleaningClass()
        self.File_Path = "C:\\Users\\shind\\JupiterWorking\\iNuron\\EDA\\Data Travel\\Data_Train.xlsx"
        self.folder_path = "artifacts/raw_Data_ingestion/"

    def dataIngestionMethod(self):
        try:
            
            df = self.preprocessingObj.readingDataSet(self.File_Path)
            self.preprocessingObj.dataIngestion(df,self.folder_path)

            print(df)

            return df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    dataIngestionClass = dataIngestionClass()
    dataIngestionClass.dataIngestionMethod()
