import os
import pickle
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path: str, obj) -> None:
    """
    Menyimpan objek Python ke file menggunakan pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Objek berhasil disimpan ke: {file_path}")

    except Exception as e:
        logging.error(f"Gagal menyimpan objek ke {file_path}")
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Memuat objek Python dari file pickle.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Objek berhasil dimuat dari: {file_path}")
        return obj

    except Exception as e:
        logging.error(f"Gagal memuat objek dari {file_path}")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict) -> dict:
    try:
        report = {}
        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")

            param_grid = params.get(model_name, {})
            gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)

            score = r2_score(y_test, y_pred)
            report[model_name] = {
                "model": best_model,
                "r2_score": score,
                "best_params": gs.best_params_
            }

        return report

    except Exception as e:
        logging.error("Gagal melakukan evaluasi model")
        raise CustomException(e, sys)