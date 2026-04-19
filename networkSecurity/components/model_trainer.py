from networkSecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networkSecurity.entity.config_entity import ModelTrainerConfig
from networkSecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from networkSecurity.utils.ml_utils.model.estimator import NetworkModel
from networkSecurity.utils.ml_utils.metric.classification_metric import get_classification_score

 
import numpy as np
import pandas as pd
import os
import sys
import mlflow
import dagshub
dagshub.init(repo_owner='LF4768', repo_name='Project_NetworkSecurity', mlflow=True)



from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier



from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging


class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def track_mlflow(self,best_model, classification_metric):
        try:
            with mlflow.start_run():
                f1_score = classification_metric.f1_score
                precision_score = classification_metric.precision_score
                recall_score = classification_metric.recall_score

                mlflow.log_metric("f1_score", f1_score)
                mlflow.log_metric("precision_score", precision_score)
                mlflow.log_metric("recall_score", recall_score)
                mlflow.sklearn.log_model(best_model, "model")
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def train_model(self,x_train,y_train, x_test, y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boost": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
            params = {
                "Random Forest": {
                    "n_estimators" : [8,16,32,64,128,256],
                    # "criterion" : ['gini', 'entropy', 'log_loss'],
                    # "max_depth":[None,2,5,10],
                },
                "Decision Tree": {
                    "criterion": ['gini', 'entropy', 'log_loss'],
                    # "max_depth": [None,2,5,10],
                    # "min_samples_split": [2,3,4,5,10]
                },
                "Gradient Boost": {
                    # "loss": ['log_loss', 'deviance', 'exponential'],
                    "learning_rate": [0.1,0.01,0.001],
                    "n_estimators": [8,16,32,64,128,256],
                    # "max_depth": [1,3,4,5,10],   
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    "n_estimators": [8,16,32,64,128,256],
                    "learning_rate": [1,0.1,0.01,0.001],
                }
            }

            model_report, params_report = evaluate_models(x_train,y_train,x_test,y_test,models,params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_params = params_report[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            
            return (best_model, best_params)
            

        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            model, params = self.train_model(x_train,y_train,x_test,y_test)
            model.set_params(**params)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            classification_train_metric = get_classification_score(y_train,y_train_pred)
            classification_test_metric = get_classification_score(y_test,y_test_pred)

            self.track_mlflow(model,classification_train_metric)
            self.track_mlflow(model,classification_test_metric)

            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=model)
            save_object(self.model_trainer_config.trained_model_file_path, network_model)

            save_object("final_models/model.pkl", model)

            model_trainer_artifact:ModelTrainerArtifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact= classification_train_metric,
                test_metric_artifact= classification_test_metric
            )

            logging.info(f"Model trainer Artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)