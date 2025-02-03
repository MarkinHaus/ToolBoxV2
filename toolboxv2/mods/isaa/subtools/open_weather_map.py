
# pip install --upgrade --quiet  pyowm
import os
from langchain_community.document_loaders import WeatherDataLoader

_api_key_ = lambda: os.environ.get('OPENWEATHERMAP')


def get_weather_data(places: list, api_key=_api_key_()):
    if isinstance(places, str):
        places = [places]

    loader = WeatherDataLoader.from_params(places=places, openweathermap_api_key=api_key)
    docs = lambda: loader.load()
    return loader, docs

