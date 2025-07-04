import sys
import pandas as pd
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging
import os

class PredictPipeline:
    def __init__(self):
        try:
            self.model_path = os.path.join("artifacts", "model.pkl")
            self.preprocessor_path = os.path.join("artifacts", "preprocessing.pkl")
            self.model = load_object(self.model_path)
            self.preprocessor = load_object(self.preprocessor_path)
            logging.info("Model dan preprocessor berhasil dimuat")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            transformed_features = self.preprocessor.transform(features)
            predictions = self.model.predict(transformed_features)
            return predictions
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            df = pd.DataFrame(data)
            logging.info("Data berhasil diubah menjadi DataFrame")
            return df
        except Exception as e:
            raise CustomException(e, sys)
