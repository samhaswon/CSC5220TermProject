import argparse
from dataclasses import dataclass
import requests
from typing import Union

DEFAULT_URL = "https://developer.nrel.gov/api/routee/v3/powertrain/route"
DEFAULT_KEY = {"api_key": "DEMO_KEY"}


class Request:
    
    def __init__(self, model: str, id: int, miles: int, speed_mph: Union[int, float], grade_percent: int) -> None:
        self.car_model         = model
        self.id                = id
        self.miles             = miles
        self.speed_mph         = speed_mph
        self.grade_percent     = grade_percent
        self.api_key:      str = "DEMO_KEY"


    def load_api_key(self, key_file: str) -> None:
        '''
        
        '''
        with open(key_file, 'r') as file:
            self.api_key = file.readline()
            if(self.api_key[-1] == '\n'):
                self.api_key = self.api_key[-1]
        # HACK: Request class needs it's own key store
        DEFAULT_KEY["api_key"] = self.api_key


    def make_request(self) -> requests.Response:
        '''
        
        '''
        data = {"powertrain_model": self.car_model,
                "links":
                    [{"id": self.id,
                      "miles": self.miles,
                      "speed_mph": self.speed_mph,
                      "grade_percent": self.grade_percent}]
                }
        return requests.post(DEFAULT_URL, params=DEFAULT_KEY, json=data)




if __name__ == "__main__":
    vehicle_request = Request(model="2016_TOYOTA_Corolla_4cyl_2WD", id=1, miles=20, speed_mph=40, grade_percent=10)
    response = vehicle_request.make_request()
    print(response.text)
