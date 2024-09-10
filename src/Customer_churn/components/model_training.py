# import os
# import sys
# import json
# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, classification_report


# from dataclasses import dataclass

# from sklearn.metrics import r2_score
# # from catboost import CatBoostClassifier
# from sklearn.ensemble import (
#     AdaBoostClassifier,
#     GradientBoostingClassifier,
#     RandomForestClassifier,
# )
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier

# from Customer_churn.exception import CustomException
# from Customer_churn.logging import logging
# from Customer_churn.utils.common import save_object, evaluate_model
# from Customer_churn.entity.config_entity import ModelTrainerConfig


# class ModelTrainer:
#     def __init__(self,config: ModelTrainerConfig):
#         self.model_trainer_config = config

#     def initiate_model_trainer(self, train_array, test_array):
#         try:
#             logging.info("Split training and test input data")

#             X_train, y_train, X_test, y_test = (
#                 train_array[:,:-1],
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1]
#             )

#             models = {
#                 "Random Forest": RandomForestClassifier(),
#                 "Decision Tree": DecisionTreeClassifier(),
#                 "Gradient Boosting": GradientBoostingClassifier(),
#                 "XG Boost": XGBClassifier(),
#                 # "CatBoosting Classifier": CatBoostClassifier(verbose=False),
#                 "AdaBoost Classifier": AdaBoostClassifier(),
#             }
#             ## Hyperparameter Tuning
#             # params = {
#             #     "Decision Tree": {
#             #         'criterion': ['gini', 'entropy'],  # Suitable for classification
#             #         'splitter': ['best', 'random'],
#             #         'max_depth': [None, 10, 20],
#             #         'min_samples_split': [2, 5],
#             #         'min_samples_leaf': [1, 2, 4]
#             #     },
#             #     "Random Forest": {
#             #         'n_estimators': [50, 100],
#             #         'criterion': ['gini', 'entropy'],
#             #         'max_depth': [None, 10, 20],
#             #         'min_samples_split': [2, 5],
#             #         'min_samples_leaf': [1, 2, 4]
#             #     },
#             #     "Gradient Boosting": {
#             #         'learning_rate': [0.01, 0.05, 0.1],
#             #         'n_estimators': [50, 100],
#             #         'subsample': [0.6, 0.8, 1.0],
#             #         'max_depth': [3, 5, 7],
#             #         'min_samples_split': [2, 5, 10],
#             #         'min_samples_leaf': [1, 2, 4]
#             #     },
#             #     "XG Boost": {
#             #         'learning_rate': [0.01, 0.05, 0.1],
#             #         'n_estimators': [50, 100],
#             #         'max_depth': [3, 5, 7],
#             #         'min_child_weight': [1, 3, 5],
#             #         'subsample': [0.6, 0.8, 1.0],
#             #         'colsample_bytree': [0.6, 0.8, 1.0]
#             #     },
#             #     # "CatBoosting Classifier": {
#             #     #     'depth': [4, 6, 8],
#             #     #     'learning_rate': [0.01, 0.05, 0.1],
#             #     #     'iterations': [100, 200, 500],
#             #     #     'l2_leaf_reg': [3, 5, 7]
#             #     # },
#             #     "AdaBoost Classifier": {
#             #         'n_estimators': [50, 100],
#             #         'learning_rate': [0.01, 0.1, 1.0],
#             #         'algorithm': ['SAMME', 'SAMME.R']
#             #     }
#             # }

#             params = {
#                 "Decision Tree": {
#                     'criterion': ['gini','entropy']
#                 },
#                 "Random Forest": {
#                     'n_estimators': [50],
#                     'criterion': ['gini','entropy']
#                 },
#                 "Gradient Boosting": {
#                     'learning_rate': [0.01,0.05]
#                 },
#                 "XG Boost": {
#                     'learning_rate': [0.01,0.05]
#                 },
#                 # "CatBoosting Classifier": {
#                 #     'depth': [4, 6, 8],
#                 #     'learning_rate': [0.01, 0.05, 0.1],
#                 #     'iterations': [100, 200, 500],
#                 #     'l2_leaf_reg': [3, 5, 7]
#                 # },
#                 "AdaBoost Classifier": {
#                     'n_estimators': [50,100]
#                 }
#             }
#             model_report: dict = evaluate_model(X_train,y_train, X_test, y_test, models,param=params)

#             # To get best model score from dict
#             best_model_score = max(sorted(model_report.values()))

#             # To get best model name from dict
#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]

#             best_model = models[best_model_name]

#             # if best_model_score < 0.4:
#             #     raise CustomException("No best model found")

#             logging.info(f"Best model found on both training and testing dataset")
#             logging.info(f"The Best model is {best_model_name} ")

#             save_object(
#                 file_path=self.model_trainer_config.model_path,
#                 obj=best_model
#             )

#             predicted = best_model.predict(X_test)

#             accuracy = accuracy_score(y_test, predicted)

#             # Optionally, generate a classification report for more detailed metrics
#             report = classification_report(y_test, predicted)
#             logging.info(f"Classification Report:\n{report}")

#             return accuracy

#         except Exception as e:
#             raise CustomException(e, sys)


# modeltrainning.py

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from Customer_churn.exception import CustomException
from Customer_churn.logging import logging
from Customer_churn.utils.common import save_object, evaluate_model
from Customer_churn.entity.config_entity import ModelTrainerConfig

# MLflow and DagsHub integration
import dagshub
import mlflow

dagshub.init(repo_owner='ArafetMarnissi', repo_name='Customer_churn', mlflow=True)

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.model_trainer_config = config

    def initiate_model_trainer(self):
        try:
            train_arr=np.load(self.model_trainer_config.train_data_path)
            test_arr=np.load(self.model_trainer_config.test_data_path)
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XG Boost": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini','entropy']
                },
                "Random Forest": {
                    'n_estimators': [50,60],
                    'criterion': ['gini','entropy']
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01,0.05]
                },
                "XG Boost": {
                    'learning_rate': [0.01,0.05]
                },
                "AdaBoost Classifier": {
                    'n_estimators': [50,100]
                }
            }

            with mlflow.start_run():
                # mlflow.autolog()
                model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models, param=params)

                # Log model parameters
                for model_name, param_set in params.items():
                    mlflow.log_param(f'{model_name}_params', param_set)

                # To get best model score from dict
                best_model_score = max(sorted(model_report.values()))

                # To get best model name from dict
                best_model_name = list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
                ]

                best_model = models[best_model_name]

                logging.info(f"Best model found on both training and testing dataset")
                logging.info(f"The Best model is {best_model_name}")

                # Save the best model object
                save_object(
                    file_path=self.model_trainer_config.model_path,
                    obj=best_model
                )

                # Make predictions
                predicted = best_model.predict(X_test)

                # Calculate accuracy
                accuracy = accuracy_score(y_test, predicted)
                mlflow.log_metric('accuracy', accuracy)

                # Optionally, generate a classification report for more detailed metrics
                report = classification_report(y_test, predicted, output_dict=True)
                mlflow.log_metric('precision', report['weighted avg']['precision'])
                mlflow.log_metric('recall', report['weighted avg']['recall'])
                mlflow.log_metric('f1-score', report['weighted avg']['f1-score'])

                logging.info(f"Classification Report:\n{classification_report(y_test, predicted)}")

                return accuracy

        except Exception as e:
            raise CustomException(e, sys)
