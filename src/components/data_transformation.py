import os
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
import numpy as np

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join("artifacts", "preprocessing.pkl")

    def get_preprocessor(self) -> ColumnTransformer:
        try:
            # Kolom input
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            numerical_columns = ["writing_score", "reading_score"]

            # Pipeline untuk numerik
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])

            # Pipeline untuk kategorikal
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            # Gabungkan keduanya
            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            logging.error("Gagal membuat preprocessor")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Memulai proses data transformation")

            preprocessor = self.get_preprocessor()
            target_column="math_score"

            # Pisahkan fitur dan target
            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            # Transformasi fitur
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Gabungkan fitur dan target
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            # Simpan preprocessor
            save_object(self.preprocessor_obj_file_path, preprocessor)
            logging.info(f"Preprocessor disimpan di: {self.preprocessor_obj_file_path}")

            return train_arr, test_arr, self.preprocessor_obj_file_path
        
        except Exception as e:
            logging.error("Gagal melakukan transformasi data")
            raise CustomException(e, sys)
