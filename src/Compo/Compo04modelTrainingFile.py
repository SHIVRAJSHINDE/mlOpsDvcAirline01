import os
import sys
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor


import warnings
import pandas as pd
#import dill
import pickle



from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.svm import SVR



from src.ExceptionLoggerAndUtils.exception import CustomException
from src.ExceptionLoggerAndUtils.utils import save_object, evaluate_Regression_models
from src.ExceptionLoggerAndUtils.logger import App_Logger


#from xgboost import XGBRegressor



class modelTrainingClass:
    def __init__(self):
            self.log_writer = App_Logger()
            self.trained_model_file_path = os.path.join("artifacts", "model.pkl")

    def modelsToTrainAndParameters(self):
        try:
            models = {
                "Random Forest"         : RandomForestRegressor(),
                "Gradient Boosting"     : GradientBoostingRegressor(),
                "Linear Regression"     : LinearRegression(),
                "lasso"                 : Lasso(),
                "ridge"                 : Ridge(),
                "AdaBoost Regressor"    : AdaBoostRegressor(),
                "SVR"                   : SVR()

            }

            params = {
                    "Random Forest": {
                        'n_estimators': [10, 50, 100, 200],  # Number of trees in the forest
                        'max_depth': [None, 10, 20, 30],   # Maximum depth of each tree
                        'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
                        'min_samples_leaf': [1, 2, 4],    # Minimum samples required in a leaf node
                        'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider
                    },
                        "Gradient Boosting": {
                        'n_estimators': [100, 200, 300],  # Number of boosting stages (trees)
                        'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage to prevent overfitting
                        'max_depth': [3, 4, 5],  # Maximum depth of each tree
                        'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
                        'min_samples_leaf': [1, 2, 4],  # Minimum samples required in a leaf node
                        'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider
                    },
                    "Linear Regression": {
                        #'alpha': [0.01, 0.1, 1.0, 10.0],   # Regularization strength (alpha)
                        'fit_intercept': [True, False]      # Whether to fit the intercept
                    },
                    "lasso": {
                        'alpha': [0.01, 0.1, 1.0, 10.0],    # Regularization strength (alpha)
                        'fit_intercept': [True, False]      # Whether to fit the intercept
                    },
                    "ridge": {
                        'alpha': [0.01, 0.1, 1.0, 10.0],  # Regularization strength (alpha)
                        'fit_intercept': [True, False]  # Whether to fit the intercept
                    },
                    "AdaBoost Regressor":{
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.1, 0.01, 0.001],
                        'base_estimator': [DecisionTreeRegressor(max_depth=3), DecisionTreeRegressor(max_depth=5)],
                        'loss': ['linear', 'square', 'exponential']
                    },
                    "SVR":{
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    },
                }

            return models , params
        except Exception as e:
            raise CustomException(e, sys)


    def modelTraingMethod(self,model_report,models):
        try:


            ## below returns the best model score from dict
            best_model_score = max(sorted(model_report.values()))
            ## below returns the best model Name score from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            ## To get best model name from dict
            best_model = models[best_model_name]

            print("Best found model on both training and testing dataset")
            print(best_model)

            save_object(
                file_path=self.trained_model_file_path,
                obj=best_model
            )


        except Exception as e:
            raise CustomException(e, sys)


    def evaluate_Regression_models(self,X_train, y_train, X_test, y_test, models, param):
        try:

            modelScore = {'modelName'   :[],
                          'R2core'      :[],
                          'aR2'         :[],
                          'MSE'         :[],
                          'MAE'         :[],
                          'RMSE'        :[]
                          }

            ar2Score = {}

            j = 0
            k = 0
            print(j)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)  # Ignore the UserWarning

                for i in range(len(list(models))):
                    print(j)
                    j = j + 1
                    model = list(models.values())[i]

                    print(list(models.keys())[i])

                    para = param[list(models.keys())[i]]
                    print(para)

                    gs = RandomizedSearchCV(model, para)
                    gs.fit(X_train, y_train)

                    model.set_params(**gs.best_params_)

                    model.fit(X_train, y_train)

                    y_train_pred = model.predict(X_train)

                    y_test_pred = model.predict(X_test)

                    R2train_model_score = r2_score(y_train, y_train_pred)
                    R2test_model_score = r2_score(y_test, y_test_pred)

                    aR2 = 1 - (1 - R2test_model_score) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

                    MSE = metrics.mean_squared_error(y_test, y_test_pred)
                    MAE = metrics.mean_absolute_error(y_test, y_test_pred)
                    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

                    ar2Score[list(models.keys())[i]] = aR2

                    modelName = list(models.keys())[i]

                    modelScore['modelName'].append(modelName)
                    modelScore['R2core'].append(R2test_model_score)
                    modelScore['aR2'].append(aR2)
                    modelScore['MSE'].append(MSE)
                    modelScore['MAE'].append(MAE)
                    modelScore['RMSE'].append(RMSE)
                    k = k+1


            print(pd.DataFrame(modelScore))
            #print(modelScore)
            print(ar2Score)

            return ar2Score

        except Exception as e:
            raise CustomException(e, sys)


