import sys
import pandas as pd
from ExceptionLoggerAndUtils.logger import App_Logger
from ExceptionLoggerAndUtils.exception import CustomException
from ExceptionLoggerAndUtils.utils import load_object


class predictionDataPreprocessingClass():

        def convertDataToDataFrame(self,Airline,Date_of_Journey,Source,Destination,
                                                          Dep_Time,Arrival_Time,Duration,Total_Stops):
            inputDict = {
                "Airline": [Airline][0],
                "Date_of_Journey": [Date_of_Journey][0],
                "Source": [Source][0],
                "Destination": [Destination][0],
                "Dep_Time": [Dep_Time][0],
                "Arrival_Time": [Arrival_Time][0],
                "Duration": [Duration][0],
                "Total_Stops": [Total_Stops][0]

            }

            df = pd.DataFrame(inputDict)
            print("----df----")
            print(df.T)
            return df

        def changeDatatypeOfColumn(self,pred_df):

            """ Written By  : Shivraj Shinde//Version: 1.0//Revisions: None
                Description : This will convert date to proper datetime format.
                Output      : Return dataFrame with proper datetime format of Date_of_Journey
                On Failure  : Raise Exception
            """
            try:
                date_format = "%Y-%m-%d %H:%M:%S"
                print(pred_df['Date_of_Journey'])
                pred_df['Date_of_Journey'] = pd.to_datetime(pred_df['Date_of_Journey'], format=date_format)
                pred_df['Dep_Time'] = pd.to_datetime(pred_df['Dep_Time'], format=date_format)
                pred_df['Arrival_Time'] = pd.to_datetime(pred_df['Arrival_Time'], format=date_format)

                return pred_df
            except Exception as e:
                    raise CustomException(e, sys)


        def convertDateInToDayMonthYear(self, df):
            """ Written By  : Shivraj Shinde//Version: 1.0//Revisions: None
                Description : This will create three different columns Day,Month,Year.
                Output      : Return dataFrame with independant Column as Day,Month,Year Columns
                On Failure  : Raise Exception
            """
            try:
                df['Day'] = pd.to_datetime(df["Date_of_Journey"], format="%Y-%m-%d").dt.day
                df['Month'] = pd.to_datetime(df['Date_of_Journey'], format="%Y-%m-%d").dt.month
                df['Year'] = pd.to_datetime(df['Date_of_Journey'], format="%Y-%m-%d").dt.year

                return df

            except Exception as e:
                raise CustomException(e, sys)

        def isertValueInDuration(self, pred_df):
            """ Written By  : Shivraj Shinde//Version: 1.0//Revisions: None
                Description : This will insert hours and minutes sting value in Duration Column.
                Output      : Return dataFrame with hours and minutes string value in Duration Column
                On Failure  : Raise Exception
            """

            pred_df['Duration'] = pred_df['Arrival_Time'] - pred_df['Dep_Time']
            #hours = str()
            Hours = pd.to_datetime(pred_df['Duration']).dt.hour
            Minutes = pd.to_datetime(pred_df['Duration']).dt.minute
            print(Hours)
            print(Hours)

            pred_df['Duration'] = pred_df['Duration'].astype(str)
            pred_df['Duration'] = str(Hours[0])+"h "+str(Minutes[0])+"m"

            return pred_df

        def dropUncessaryColumns(self,df):
            """ Written By  : Shivraj Shinde//Version: 1.0//Revisions: None
                Description : This will drop irrelevent columns from dataFrame
                Output      : Return dataFrame where irrelevent columns are deleted.
                On Failure  : Raise Exception
            """

            try:
                df = df.drop(["Arrival_Time","Dep_Time","Date_of_Journey","Duration"], axis = 1)
                return df
            except Exception as e:
                raise CustomException(e, sys)

