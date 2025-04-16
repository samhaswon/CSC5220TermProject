"""
Estimates route fuel usage with NREL's RouteE API
"""
import datetime
import json
import os
from time import sleep
from typing import List

import pandas as pd
from tqdm import tqdm
# pylint: disable=import-error
import NREL_API


# ================================================================================================
def load_data(data_dir: str = "./cleaned_data") -> List[pd.DataFrame]:
    """
    Loads the dataset.
    """
    # Adapted from Sam's code in data_loading/vehicle_dataset.py
    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    dfs = [pd.read_csv(f) for f in tqdm(file_list, desc="Loading Data Files")]
    # Trim space from column names
    for df in dfs:
        df.columns = df.columns.str.strip()
    return dfs


def calculate_duration(data: List[pd.DataFrame]) -> List[datetime.timedelta]:
    """
    Calculates the duration of a trip
    """
    trip_durations = []
    for frame in data:
        # What a nightmare this is : )
        start_time = frame["GPS Time"].head(1).iloc(0)[0]
        end_time = frame["GPS Time"].tail(1).iloc(0)[0]
        start_time = start_time.split(' ')
        parsed_time = start_time[3].split(':')
        start_time = datetime.datetime(year=int(start_time[5]),
                                       month=datetime.datetime.strptime(
                                           start_time[1], "%b").month,
                                       day=int(start_time[2]),
                                       hour=int(parsed_time[0]),
                                       minute=int(parsed_time[1]),
                                       second=int(parsed_time[2]))
        end_time = end_time.split(' ')
        parsed_time = end_time[3].split(':')
        end_time = datetime.datetime(year=int(end_time[5]),
                                     month=datetime.datetime.strptime(end_time[1], "%b").month,
                                     day=int(end_time[2]),
                                     hour=int(parsed_time[0]),
                                     minute=int(parsed_time[1]),
                                     second=int(parsed_time[2]))
        trip_durations.append(end_time - start_time)
    return trip_durations


def get_averages(data: List[pd.DataFrame], parameter: str) -> List[float]:
    """
    Averages the relevant metrics of the data.
    """
    avgs = [frame[parameter].mean().item() for frame in data]
    return avgs


def calculate_distance(avg_speeds: List[float], trip_durations: List[datetime.timedelta]) -> List[float]:
    """
    Calculates the distance of the trip.
    """
    miles = []
    for i in range(0, len(avg_speeds)):
        miles.append(avg_speeds[i] * (trip_durations[i].total_seconds() / 3600))
    return miles


def make_requests(
        request: NREL_API.Request, grades: List[float], speeds: List[float],
        trip_duration: List[datetime.timedelta], time_delay: int = 5) -> list[str]:
    """
    Makes the requests to NREL's RouteE API
    """
    miles = calculate_distance(speeds, trip_duration)
    text_list = []
    for i in tqdm(range(0, len(grades)), desc="Requesting RouteE Data"):
        sleep(time_delay)
        request.miles = miles[i]
        request.speed_mph = speeds[i]
        request.grade_percent = grades[i]
        text_list.append({'routee_estimate': request.make_request().json()["route"][0]['energy_estimate']})
    return text_list


# ================================================================================================
if __name__ == "__main__":
    dataframes = load_data()
    for i, dataframe in enumerate(dataframes):
        cols = list(dataframe.columns)
        assert "Grade" in cols, i
    grade_averages = get_averages(dataframes, "Grade")
    speed_averages = get_averages(dataframes, "Speed (OBD)(mph)")
    durations = calculate_duration(dataframes)
    req = NREL_API.Request(
        model="2016_TOYOTA_Corolla_4cyl_2WD",
        id=1, miles=20, speed_mph=40, grade_percent=10
    )
    req.load_api_key("./RouteE.key")
    responses = make_requests(req, grade_averages, speed_averages, durations, time_delay=0)
    with open("estimates.json", 'w', encoding="utf-8") as results:
        json.dump(responses, results, indent=4)
