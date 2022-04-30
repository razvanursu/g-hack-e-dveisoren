from datetime import datetime, timedelta, timezone
from meteostat import Point, Daily, Hourly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import os as os
from sqlalchemy import create_engine


class linear_model:
    """linear regression model
    """

    def __init__(self):
        try:
            self.weights = np.load((Path.cwd() / "energygen" / "predictor" / "linare_weights.npy").resolve())
        except:
            print("no weights found")

    def inference(self, input_array):
        """ predict renewable engery percentage with a linear model:

        Args:
            input_array (np.array): n x f - array (n samples and f features)

        Return:
            output_array (np.array): n - array (n samples)
        """
        return input_array @ self.weights

    def train(self, X, Y):
        """trains linear model and saves the learned parameters

        Args:
            X (np.array): feature-array: n x f - array (n samples and f features)
            Y (np.array): label-array: n x 1 - array (n samples)
        """
        print("Training...")
        print("X-shape: ", X.shape)
        print("Y-shape: ", Y.shape)

        self.weights = np.linalg.pinv(X) @ Y

        print("trained linear weights: ", self.weights)
        np.save("linare_weights.npy", self.weights)


def load_historic_weather():
    """load historic weather data for training our model

    Returns:
        np.array: historic weather data for the United Kingdom between 2018-2021
    """
    data = pd.DataFrame()
    for year in [2018, 2019, 2020, 2021]:
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23)
        location = Point(51, -0.1277)
        year_data = Hourly(location, start, end)
        year_data = year_data.fetch()
        data = data.append(year_data)
    data["hour"] = data.index.hour
    data["month"] = data.index.month
    data = data.drop("tsun", axis=1)
    data = data.drop("snow", axis=1)
    data = data.drop("coco", axis=1)
    data = data.drop("wpgt", axis=1)
    data = data.drop("prcp", axis=1)
    data = data.drop("wdir", axis=1)
    # print(data.columns)
    return data.to_numpy()


def load_historic_renewability():
    """ load historic renewability percentage for training our model

    Returns:
        np.array: historic renewability percentage data for the United Kingdom between 2018-2021
    """
    db_user = os.getenv('DATABASE_USER')
    db_pass = os.getenv('DATABASE_PASSWORD')
    # db_name = os.getenv('DATABASE_NAME')
    # db_connection_name = os.environ.get('DATABASE_NAME_CONNECTION_NAME')
    engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_pass}@localhost:5432/')
    brit = pd.read_sql_table('select * from "fuel";', con=engine)
    brit = brit[::2]
    brit = brit[brit['DATETIME'] >= "2018-01-01 00:00:00"]
    brit = brit[brit['DATETIME'] < "2022-01-01 00:00:00"]
    Y = (brit["RENEWABLE_perc"]).to_numpy()
    return np.array([Y]).T


def get_weather_features_of_location(gps1, gps2, timezone_offset=2):
    """Get weater forecast of next 24 hours at given location

    Args:
        gps1 (float): gps longitude
        gps2 (float): gps latitude
        timezone_offset (int): offset of timezone. Defaults to 2.

    Returns:
        np.array: weather-features in numpy array
    """
    now = datetime.now(timezone(timedelta(hours=timezone_offset)))

    tomorrow = now + timedelta(days=1) - timedelta(hours=1)

    start = datetime(now.year, now.month, now.day, now.hour)
    end = datetime(tomorrow.year, tomorrow.month, tomorrow.day, tomorrow.hour)
    location = Point(gps1, gps2)
    data = Hourly(location, start, end)
    data = data.fetch()

    data["hour"] = data.index.hour
    data["month"] = data.index.month
    data = data.drop("tsun", axis=1)
    data = data.drop("snow", axis=1)
    data = data.drop("wpgt", axis=1)
    data = data.drop("coco", axis=1)
    data = data.drop("prcp", axis=1)
    data = data.drop("wdir", axis=1)
    # print(data)
    return data.to_numpy()


def get_country_renewability(country="United Kingdom"):
    """ get mean percentage of renewables in electricity grid of given country

    Args:
        country (str): Country Name. Defaults to "United Kingdom".

    Returns:
        float: percentage of renewables in electricity grid
    """
    db_user = os.getenv('DATABASE_USER')
    db_pass = os.getenv('DATABASE_PASSWORD')
    # db_name = os.getenv('DATABASE_NAME')
    # db_connection_name = os.environ.get('DATABASE_NAME_CONNECTION_NAME')
    engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_pass}@localhost:5432/')
    data = pd.read_sql_table('select * from "elec";', con=engine)
    # data = pd.read_sql_table('elec', 'postgres:///postgres')
    data = data[data["Year"] == 2020]
    data = data[data["Entity"] == country]
    data_renew = data[["Nuclear",
                       "Hydro",
                       "Wind",
                       "Solar",
                       "Other"]]

    renew = data_renew.sum(axis=1).to_numpy()[0]
    return renew/100


def predict_renew_location(gps1, gps2, timezone_offset, country):
    """ predict local renewable percentage for 24 hours

    Args:
        gps1 (float): gps longitude
        gps2 (float): gps latitude
        timezone_offset (int): offset of timezone.
        country (String): Country Name

    Returns:
        np.array: 24 hourly predictions of local renewable percentage
    """
    weather = get_weather_features_of_location(gps1, gps2, timezone_offset)
    prediction_uk = linear_model().inference(weather)

    # calculate offset bewteen United Kingdom and the seletcted Country
    # (the model was only trained on date from UK)
    prediction_final = prediction_uk * \
        get_country_renewability(country) / \
        get_country_renewability("United Kingdom")
    return prediction_final.T[0]


if __name__ == "__main__":

    """TRAINING"""
    # model = linear_model()
    # model.train(load_historic_weather(), load_historic_renewability())

    """PLOTTING"""
    weather = get_weather_features_of_location(48, 11.5, 2)
    plot_names = ["Temperature", "Dew Point", "Humidity",
                  "Wind-Speed", "Pressure", "Hour", "Month"]
    for i in range(7):
        plt.subplot(1, 7, i+1)
        plt.plot(weather[:, i])
        plt.title(plot_names[i])
    plt.show()

    pred = predict_renew_location(34, -118, -7, "United States")
    plt.plot(pred, color='b')
    plt.axhline(y=np.mean(pred), color='b', linestyle='-.')
    pred = predict_renew_location(48, 11.5, 2, "Germany")
    plt.plot(pred, color='r')
    plt.axhline(y=np.mean(pred), color='r', linestyle='-.')
    pred = predict_renew_location(51, -0.1277, 1, "United Kingdom")
    plt.plot(pred, color='g')
    plt.axhline(y=np.mean(pred), color='g', linestyle='-.')
    plt.legend(["Los Angeles", "mean", "Munich", "mean", "London", "mean"])
    plt.ylabel("% of renewable electricity")
    plt.xlabel("hours into future")
    plt.title("Our Prediction for next 24 hours Renewablity at 3 Locations")
    plt.show()
