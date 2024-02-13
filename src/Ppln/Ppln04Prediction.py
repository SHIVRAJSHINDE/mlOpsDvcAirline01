import pandas as pd
from Pipeline.Pipeline03Prediction.predictionDataPreprocessingFile import predictionDataPreprocessingClass
from Pipeline.Pipeline01Preprocessing.dataReadingAndCleaningFile import dataReadingAndCleaningClass
from Pipeline.Pipeline03Prediction.predictionFile import predictionClass

class predictionInitiatorClass():
    def __init__(self):

        self.predictionDataPreprocessingObj = predictionDataPreprocessingClass()
        self.dataReadingAndCleaningObj = dataReadingAndCleaningClass()
        self.predictionObj = predictionClass()

    def receiveDataFromUI(self,Airline:str,Date_of_Journey:str,Source:str,Destination:str,
                     Dep_Time:str,Arrival_Time:str,Duration:str,Total_Stops:str):

        pred_df = self.predictionDataPreprocessingObj.convertDataToDataFrame(Airline,Date_of_Journey,Source,Destination,
                                                                             Dep_Time,Arrival_Time,Duration,Total_Stops)

        pred_df = self.predictionDataPreprocessingObj.changeDatatypeOfColumn(pred_df)
        pred_df = self.predictionDataPreprocessingObj.convertDateInToDayMonthYear(pred_df)
        pred_df = self.dataReadingAndCleaningObj.convertHoursAndMinutesToIndependantColumns(df=pred_df, columName="Dep_Time")
        pred_df = self.dataReadingAndCleaningObj.convertHoursAndMinutesToIndependantColumns(df=pred_df, columName="Arrival_Time")
        pred_df = self.predictionDataPreprocessingObj.isertValueInDuration(pred_df)
        print(pred_df.T)
        pred_df = self.dataReadingAndCleaningObj.convertDurationToMunutes(pred_df)
        pred_df  = self.predictionDataPreprocessingObj.dropUncessaryColumns(pred_df)
        output = self.predictionObj.predictionMethod(pred_df)


        return output
