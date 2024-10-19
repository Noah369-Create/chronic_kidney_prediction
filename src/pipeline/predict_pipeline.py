import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        blood_urea: float,
        white_blood_cell_count:float,
        blood_glucose_random: float,
        serum_creatinine: float,
        albumin: float,
        hypertension: float,
        ):

        self.blood_urea=blood_urea        
        self.white_blood_cell_count=white_blood_cell_count
        self.blood_glucose_random=blood_glucose_random
        self.serum_creatinine=serum_creatinine
        self.albumin=albumin
        self.hypertension = hypertension
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "blood_urea": [self.blood_urea],
                "white_blood_cell_count": [self.white_blood_cell_count],
                "blood_glucose_random": [self.blood_glucose_random],
                "serum_creatinine": [self.serum_creatinine],
                "albumin": [self.albumin],
                "hypertension":[self.hypertension]
                
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)