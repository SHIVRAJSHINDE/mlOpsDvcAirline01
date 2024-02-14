import os
import sys
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

from src.ExceptionLoggerAndUtils.exception import CustomException
from src.ExceptionLoggerAndUtils.utils import save_object, evaluate_Regression_models
from src.ExceptionLoggerAndUtils.logger import App_Logger


import warnings
import pandas as pd
#import dill
import pickle

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.svm import SVR




#from xgboost import XGBRegressor



class modelTrainingClass:
    def __init__(self):
            '''self.log_writer = App_Logger()'''
            self.trained_model_file_path = os.path.join("artifacts", "model.pkl")



    def mlFlowParametersAndModelsForTraining(self):
        try:
            models = {
                "Random Forest"         : RandomForestRegressor(),

                }

            params = {
                    "Random Forest": {
                        'n_estimators': [10, 50, 100, 200],  # Number of trees in the forest
                        'max_depth': [None, 10, 20, 30],   # Maximum depth of each tree
                        'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
                        'min_samples_leaf': [1, 2, 4],    # Minimum samples required in a leaf node
                        'max_features': ['sqrt', 'log2', None]  # Number of features to consider
                    },

                    
                }

            return models , params
        except Exception as e:
            raise CustomException(e, sys)

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    '''csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/data/winequality-red.csv"
    )'''
    
    csv_url = "D:\\Downloads\\winequality-red.csv"
    
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        modelTrainingObj = modelTrainingClass()

        (rmse, mae, r2) = modelTrainingObj.eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

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







