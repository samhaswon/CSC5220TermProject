import NREL_API

import datetime
import os
import pandas as pd
import requests
from time import sleep
from tqdm import tqdm
from typing import List


# ======================================================================================================================
def load_data(data_dir: str = "./cleaned_data") -> List[pd.DataFrame]:
    '''
    
    '''
    # Adapted from Sam's code in data_loading/vehicle_dataset.py
    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    dataframes = [pd.read_csv(f) for f in tqdm(file_list, desc="Loading Data Files")]
    return dataframes


def calculate_duration(data: List[pd.DataFrame]) -> List[datetime.timedelta]:
    '''
    
    '''
    durations = []
    for frame in data:
        # What a nightmare this is : )
        start_time = frame["GPS Time"].head(1).iloc(0)[0]
        end_time   = frame["GPS Time"].tail(1).iloc(0)[0]
        start_time = start_time.split(' ')
        parsed_time = start_time[3].split(':')
        start_time = datetime.datetime(year=int(start_time[5]),
                                       month=datetime.datetime.strptime(start_time[1], "%b").month,
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
        durations.append(end_time - start_time)
    return durations


def get_averages(data: List[pd.DataFrame], parameter: str) -> List[float]:
    '''

    '''
    avgs = [frame[parameter].mean().item() for frame in data]
    return avgs


def calculate_distance(avg_speeds: List[float], durations: List[datetime.timedelta]) -> List[float]:
    '''
    
    '''
    miles = []
    for i in range(0, len(avg_speeds)):
        miles.append(avg_speeds[i] * (durations[i].total_seconds() / 3600))
    return miles



def make_requests(req: NREL_API.Request, grades: List[float], speeds: List[float], durations: List[datetime.timedelta], time_delay: int = 5) -> List[requests.Response]:
    '''

    '''
    miles = calculate_distance(speeds, durations)
    text_list = []
    for i in tqdm(range(0, len(grades)), desc="Requesting RouteE Data"):
        sleep(time_delay)
        req.miles = miles[i]
        req.speed_mph = speeds[i]
        req.grade_percent = grades[i]
        text_list.append(req.make_request().text)
    return(text_list)




# ======================================================================================================================
if __name__ == "__main__":
    dataframes = load_data()
    grade_averages = get_averages(dataframes, "Grade")
    speed_averages = get_averages(dataframes, "Speed (OBD)(mph)")
    durations = calculate_duration(dataframes)
    req = NREL_API.Request(model="2016_TOYOTA_Corolla_4cyl_2WD", id=1, miles=20, speed_mph=40, grade_percent=10)
    req.load_api_key("./RouteE.key")
    responses = make_requests(req, grade_averages, speed_averages, durations, time_delay=10)
    with open("estimates.json", 'w') as results:
        for r in responses:
            results.write(str(r) + ',')
