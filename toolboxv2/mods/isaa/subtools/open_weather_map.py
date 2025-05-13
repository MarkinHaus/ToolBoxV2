
# pip install --upgrade --quiet  pyowm
import os

from langchain_community.document_loaders import WeatherDataLoader


def _api_key_():
    return os.environ.get('OPENWEATHERMAP')


def get_weather_data(places: list, api_key=_api_key_()):
    if isinstance(places, str):
        places = [places]

    loader = WeatherDataLoader.from_params(places=places, openweathermap_api_key=api_key)
    def docs():
        return loader.load()
    return loader, docs

