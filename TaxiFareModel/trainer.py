from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data
from TaxiFareModel.data import clean_data
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
import mlflow
from sklearn.ensemble import RandomForestRegressor
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = "LuciaMadrid22"

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
             ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
            ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', RandomForestRegressor())
            ])

        self.pipeline = pipe

    def run(self, X_train, y_train):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(X_train, y_train)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""

        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri("https://mlflow.lewagon.ai/")
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == "__main__":

    #get_data
    df = get_data(nrows=10_000)

    # clean data
    df = clean_data(df)

    # set X and y
    y = df.pop("fare_amount")
    X = df

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # train
    model = Trainer(X,y)
    model.run(X_train, y_train)

    # evaluate
    evaluation = model.evaluate(X_test, y_test)

    model.mlflow_log_metric("rmse", evaluation)
    model.mlflow_log_param("model", "RandomForest")
