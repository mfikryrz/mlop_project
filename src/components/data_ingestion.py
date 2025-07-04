import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

import sys

class DataIngestionConfig:
    def __init__(self):
        self.raw_data_path = os.path.join('artifacts',"data.csv")
        self.train_data_path =  os.path.join('artifacts',"train.csv")
        self.test_data_path = os.path.join('artifacts',"test.csv")
        self.test_size = 0.2
        self.random_state = 42


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        logging.info("Memulai proses data ingestion")
        try:
            # Membaca data awal
            df = pd.read_csv("data/stud.csv")
            logging.info(f"Dataset berhasil dibaca, jumlah data: {df.shape}")

            # Membuat direktori untuk menyimpan data jika belum ada
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.test_data_path), exist_ok=True)

            # Simpan data mentah (raw data)
            df.to_csv(self.config.raw_data_path, index=False)
            logging.info(f"Data mentah disimpan di: {self.config.raw_data_path}")

            # Lakukan split
            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            logging.info("Split data menjadi train dan test berhasil")

            # Simpan hasil split
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logging.info(f"Data train disimpan di: {self.config.train_data_path}")
            logging.info(f"Data test disimpan di: {self.config.test_data_path}")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            logging.error("Terjadi error saat proses data ingestion")
            raise CustomException(e, sys)

if __name__=="__main__":
    obj=DataIngestion(DataIngestionConfig())
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))