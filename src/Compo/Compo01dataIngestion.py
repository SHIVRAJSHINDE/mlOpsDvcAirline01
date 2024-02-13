
from src.ExceptionLoggerAndUtils.exception import CustomException
from src.ExceptionLoggerAndUtils.logger import App_Logger

import pandas as pd
import os
import sys
import json

class dataReadingAndCleaningClass():
    """ This class shall be used for handling all the SQL operations.
        Written By: Shivraj Shinde//Version: 1.0//Revisions: None
    """

    def __init__(self):
        self.log_writer = App_Logger()
        self.cwd=os.getcwd()
        self.schema_path = "Schemas/schema_data.json"
        
    def readingDataSet(self,File_Path):
        try:

            self.df = pd.read_excel(File_Path,engine='openpyxl')
            return self.df

        except Exception as e:
            raise CustomException(e,sys)

    def dataIngestion(self,df,folder_path):
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_name = "rawData.csv"  # Name of the CSV file
            file_path = os.path.join(folder_path, file_name)  # Full file path
            df.to_csv(file_path, index=False)  # Save the DataFrame as a CSV file
            print(f"CSV file saved to {file_path}")

        except Exception as e:
            raise CustomException(e, sys)



        except Exception as e:
            raise CustomException(e, sys)

