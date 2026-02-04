from importlib import import_module

import modal
from modal import App, Image

### Setup
app = modal.App("hello")
image = Image.debian_slim().pip_install("requests")

###  Hello functions

@app.function(image=image)
def hello() -> str:
    import requests

    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    print("DATA: ", data)
    city, region, country = data["city"], data["region"], data["country"]

    return f"Hello from {city}, {region}, {country}!!"


@app.function(image=image, region="eu")
def hello_europe() -> str:
    import requests

    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    print("DATA: ", data)
    city, region, country = data["city"], data["region"], data["country"]

    return f"Hello from {city}, {region}, {country}!!"

