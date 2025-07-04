import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.utils import save_object, evaluate_models
from src.logger import logging
from src.exception import CustomException

class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Memulai proses training model")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(verbosity=0),
                "CatBoost": CatBoostRegressor(verbose=0)
            }

            params = {
                "LinearRegression": {},
                "DecisionTree": {
                    "max_depth": [3, 5, 10]
                },
                "RandomForest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10]
                },
                "GradientBoosting": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1]
                },
                "XGBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1]
                },
                "CatBoost": {
                    "depth": [4, 6],
                    "learning_rate": [0.01, 0.1]
                }
            }

            report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Pilih model terbaik berdasarkan r2_score
            best_model_name = max(report, key=lambda name: report[name]['r2_score'])
            best_model = report[best_model_name]['model']
            best_score = report[best_model_name]['r2_score']

            logging.info(f"Model terbaik: {best_model_name} dengan R²: {best_score}")
            save_object(self.model_path, best_model)

            return best_score

        except Exception as e:
            logging.error("Gagal dalam proses model training")
            raise CustomException(e, sys)
