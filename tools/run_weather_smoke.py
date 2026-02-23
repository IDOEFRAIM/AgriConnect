import sys
sys.path.insert(0, 'backend/src')
from agriconnect.services.data_collection.weather.weather_forecast import WeatherForecastService
from typing import List, Dict, Any

class MockOption:
    def __init__(self, value, label):
        self._value = value
        self._label = label
    def get_attribute(self, name):
        if name == 'value':
            return self._value
        return None
    def inner_text(self):
        return self._label

class MockLocator:
    def __init__(self, options: List[MockOption]):
        self._options = options
    def all(self):
        return self._options

class MockPage:
    def __init__(self):
        self._selected = None
        self._options = [MockOption('ouagadougou', 'Ouagadougou'), MockOption('bobo', 'Bobo-Dioulasso')]
    def wait_for_selector(self, selector, state=None, timeout=None):
        return True
    def locator(self, sel):
        return MockLocator(self._options)
    def select_option(self, selector, value):
        self._selected = value
    def wait_for_timeout(self, ms):
        return None
    def evaluate(self, script):
        # Return sample charts depending on selected city
        if self._selected == 'ouagadougou':
            return [{
                'title': 'Températures',
                'subtitle': '',
                'series': [
                    {'name': 'Max', 'data': [{'x': 1, 'y': 30}, {'x': 2, 'y': 31}]},
                    {'name': 'Min', 'data': [{'x': 1, 'y': 20}, {'x': 2, 'y': 21}]}
                ]
            }]
        elif self._selected == 'bobo':
            return [{
                'title': 'Précipitations',
                'subtitle': '',
                'series': [
                    {'name': 'Pluie', 'data': [{'x': 1, 'y': 5}]}
                ]
            }]
        return []

if __name__ == '__main__':
    service = WeatherForecastService()
    page = MockPage()
    results = service._scrape_forecasts(page)
    import json
    print(json.dumps(results, ensure_ascii=False, indent=2))
