
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformerConfig:
        preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
        def __init__(self):
                self.data_transformer_config=DataTransformerConfig()
        def get_data_transformer_object(self):
                try:

                        numeric_columns=['reading_score', 'writing_score']

                        categorical_columns =[
                                'gender',
                                'race_ethnicity',
                                'parental_level_of_education',
                                'lunch',
                                'test_preparation_course'
                                ]
                        
                        num_pipeline=Pipeline(
                                steps=[
                                        ("Imputer",SimpleImputer(strategy='median')),
                                        ("Scaler",StandardScaler())
                                ]
                        )
                        cat_pipeline=Pipeline(
                                steps=[
                                        ("Imputer",SimpleImputer(strategy="most_frequent")),
                                        ("one_hot_encoder",OneHotEncoder()),
                                        ("scaler",StandardScaler(with_mean=False))
                                ]
                        )

                        logging.info(f"Numerical Feature = {numeric_columns}")
                        logging.info(f"Categorical feature = {categorical_columns}")


                        preprocessor=ColumnTransformer(
                                [
                                        ("num_pipeline",num_pipeline,numeric_columns),
                                        ("cat_pipeline",cat_pipeline,categorical_columns)
                                ]
                        )
                        return preprocessor
                
                except Exception as e:
                        raise CustomException(e,sys)
        
        def initiate_data_transformation(self,train_path,test_path):
                try:
                        train_df=pd.read_csv(train_path)
                        test_df=pd.read_csv(test_path)

                        logging.info("Read train and test data completed")
                        logging.info("Obtaining Preprocessor object")

                        preprocessing_obj=self.get_data_transformer_object()
                        target_column_name='math_score'
                        numeric_columns=['reading_score', 'writing_score']

                        input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
                        target_feature_train_df=train_df[target_column_name]

                        input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
                        target_feature_test_df=test_df[target_column_name]

                        logging.info("Apply Preproccessing object on training dataframe and testing dataframe"    )
                        input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
                        input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
                        
                        train_arr=np.c_[
                                input_feature_train_arr,np.array(target_feature_train_df)

                        ]
                        test_arr=np.c_[
                                input_feature_test_arr,np.array(target_feature_test_df)
                        ]
                        logging.info("Saved Preprocessing Object")

                        save_object(
                                self.data_transformer_config.preprocessor_obj_file_path,
                                obj=preprocessing_obj
                        )
                        return (
                                train_arr,
                                test_arr,
                                self.data_transformer_config.preprocessor_obj_file_path,

                        )
                except Exception as e:
                        raise CustomException(e,sys)