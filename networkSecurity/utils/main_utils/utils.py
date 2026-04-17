import yaml
from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging
import os,sys
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def read_yaml_file(file_path:str) -> dict:   
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    

def write_yaml_file(file_path:str, content:object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    


def save_numpy_array_data(file_path:str, array:np.array) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise NetworkSecurityException(e,sys)

def load_numpy_array_data(file_path) -> np.array:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file{file_path} doesnt exist")
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys)

def save_object(file_path:str, obj:object)-> None:
    try:
        logging.info("Entered the save_object method of utils.py")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method")
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def load_object(file_path:str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file{file_path} doesnt exist")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj) 
    except Exception as e:
        raise NetworkSecurityException(e,sys)   
    
def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        report = {}
        best_params= []
        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(params.keys())[i]]

            gs = GridSearchCV(model,param, cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(params.keys())[i]] = test_model_score
            best_params.append(gs.best_params_)

        return (report, best_params) 
    
    except Exception as e:
        raise NetworkSecurityException(e,sys)