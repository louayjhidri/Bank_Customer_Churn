import sys 
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,FunctionTransformer
from Customer_churn.entity.config_entity import DataTransformationConfig

from Customer_churn.exception import CustomException
from Customer_churn.logging import logging
from Customer_churn.utils.common import save_object
import os
from imblearn.over_sampling import SMOTE

class DataTransformation:
    def __init__(self,config: DataTransformationConfig):
        self.config=config
        
    def create_preprocessor(self):

        numerical_columns = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                     "HasCrCard", "IsActiveMember", "EstimatedSalary",
                     "SatisfactionScore", "PointEarned"]

        categorical_columns = ["Gender", "CardType"]

        card_type_mapping = {'DIAMOND': 1, 'PLATINUM': 2, 'GOLD': 3, 'SILVER': 4}
    # Custom transformer for mapping 'CardType'
        def map_card_type(X):
            return np.vectorize(card_type_mapping.get)(X)
        
        # Pipeline for numerical columns (Standardization)
        numerical_pipeline = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Pipeline for categorical columns (OneHot Encoding and Standardization)
        categorical_pipeline = Pipeline(steps=[
            ('card_type_mapper', FunctionTransformer(map_card_type, validate=False)), # Map CardType
            ('onehot', OneHotEncoder(sparse_output=False)),                           # OneHotEncode
            ('scaler', StandardScaler())                                              # Standardize
        ])
        
        # Create a column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_columns),
                ('cat', categorical_pipeline, categorical_columns)
            ]
        )
        
        return preprocessor

    # def initiate_data_transformation(self):
    #     try:



    #         df=pd.read_csv(self.config.raw_data_path)
    #         logging.info('Read the dataset as dataframe')

    #         logging.info("Train test split initiated")
    #         train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

    #         train_set = train_set.drop(columns=['Complain'], errors='ignore')
    #         test_set = test_set.drop(columns=['Complain'], errors='ignore')

    #         train_set.to_csv(self.config.train_data_path,index=False,header=True)

    #         test_set.to_csv(self.config.test_data_path,index=False,header=True)

    #         logging.info("Ingestion of the data is completed")





    #         train_df=pd.read_csv(self.config.train_data_path)
    #         test_df=pd.read_csv(self.config.test_data_path)

    #         logging.info("Read train and test data completed")
    #         logging.info("Obtaining preprocessing object")

    #         preprocessing_obj=self.create_preprocessor()
           

    #         input_feature_train_df = train_set[['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'SatisfactionScore', 'CardType', 'PointEarned']]
    #         target_feature_train_df = train_set['Exited']

    #         input_feature_test_df = test_set[['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'SatisfactionScore', 'CardType', 'PointEarned']]
    #         target_feature_test_df = test_set['Exited']

    #         logging.info(
    #             f"Applying preprocessing object on training dataframe and testing dataframe."
    #         )
            
    #         input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
    #         input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
    #         print("train df shape = ********** : ",input_feature_train_arr.shape)
    #         print("test df shape = ********** : ",input_feature_test_arr.shape)
    #         # Apply SMOTE
    #         smote = SMOTE(sampling_strategy='auto', random_state=42)
    #         X_resampled, y_resampled = smote.fit_resample(input_feature_train_arr, target_feature_train_df)
            
    #         print("Resampled train df shape = ********** : ", X_resampled.shape)
            
    #         train_arr = np.c_[X_resampled, np.array(y_resampled)]
    #         test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            

    #         # train_arr = np.c_[
    #         #     input_feature_train_arr, np.array(target_feature_train_df)
    #         # ]
    #         # test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

    #         logging.info(f"Saved preprocessing object.")

    #         save_object(

    #             file_path=self.config.preprocessor_obj_file_path,
    #             obj=preprocessing_obj

    #         )

    #         return (train_arr,test_arr)

    #     except Exception as e:
    #         raise CustomException(e,sys)



    def initiate_data_transformation(self):
        try:
            df = pd.read_csv(self.config.raw_data_path)
            logging.info('Read the dataset as dataframe')

            # Drop 'Complain' column if present
            df = df.drop(columns=['Complain'], errors='ignore')

            logging.info("Dropping 'Complain' column completed")

            # Split features and target
            X = df[['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
                    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                    'EstimatedSalary', 'SatisfactionScore', 'CardType', 'PointEarned']]
            y = df['Exited']

            # Obtain the preprocessing object
            preprocessing_obj = self.create_preprocessor()

            # Apply preprocessing to the entire dataset
            logging.info("Applying preprocessing object on the entire dataset.")
            X_preprocessed = preprocessing_obj.fit_transform(X)

            logging.info("Preprocessing completed")

            # Apply SMOTE to handle class imbalance
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)
            
            logging.info("SMOTE resampling completed")
            print("Resampled data shape = ********** : ", X_resampled.shape)

            # Split the resampled data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
            
            # Combine features and target for train and test sets
            train_arr = np.c_[X_train, y_train]
            test_arr = np.c_[X_test, y_test]

            # Save the preprocessor object
            logging.info("Saved preprocessing object.")
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr

        except Exception as e:
            raise CustomException(e, sys)